import sys

sys.path.append('/home/delalamo/SoftAlign/')

import os
import pickle

import END_TO_END_MODELS as ete
import haiku as hk
import Input_MPNN as input_
import jax
import jax.numpy as jnp
import numpy as np
import Score_align as lddt
from Bio import PDB, SeqIO

num_layers = 3 #@param {type:"integer"}
num_neighbors = 64 #@param {type:"integer"}
encoding_dim = 64 #@param {type:"integer"}
categorical = False #@param {type:"boolean"}
nb_clusters = 100 #@param {type:"integer"}
soft_max = False #@param {type:"boolean"}
bs = 15

STRATEGIES = {
    "ALL": [f"{x}{y}" for x in range(1, 129) for y in " ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    "NO_CDR": [f"{x}{y}" for x in list(range(1, 26)) + list(range(39, 56)) + list(range(66, 105)) + list(range(118, 129)) for y in " ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    "NO_INS": [f"{x}" for x in range(1, 129)],
    "NO_CDR_NO_INS": [f"{x}" for x in list(range(1, 26)) + list(range(39, 56)) + list(range(66, 105)) + list(range(118, 129))],
    "AHO": [str(x) for x in list(range(1, 33))] + ["33A", "33B"] + [str(x) for x in list(range(34, 120))] + [f"120{x}" for x in " ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + [str(x) for x in list(range(121, 129))],
}

STRATEGIES = {k: [x.strip() for x in v] for k, v in STRATEGIES.items()}

def model_end_to_end(
        targ_arr,
        ref_arr,
        lens,
        t,
        node_features = encoding_dim,
        edge_features = encoding_dim,
        hidden_dim = encoding_dim,
        num_encoder_layers=num_layers,
        k_neighbors=num_neighbors,
        categorical = categorical,
        nb_clusters = nb_clusters,
        affine = True,
        soft_max = soft_max
    ):
    a = ete.END_TO_END(
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers,
        k_neighbors,
        affine = True,
        soft_max = soft_max,
        dropout = 0.,
        augment_eps=0.0)
    
    return a.align(targ_arr, ref_arr, lens, t)

def get_res_idx(pdb, chain):
    """
    Get the residue index for a given chain in a PDB file.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb)
    res_idx = {}
    for i, res in enumerate(structure[0][chain].get_residues()):
        if PDB.is_aa(res):
            res_idx[i] = res.get_id()[1]
    return res_idx

def calc_devs(soft_aln, targ_idxs, ref_idxs):
    devs = []
    coords = np.argwhere(np.array(soft_aln[0]) == 1)
    for i, j in coords:
        res_target = int(targ_idxs[i].split(".")[-1])
        res_ref = int(ref_idxs[j])
        if res_target != res_ref:
            devs.append((res_target, res_ref))
    return devs

def get_reps(species):
    # Load all cluster centers
    fasta_chain = species.replace("_H", "_heavy").replace("_L", "_light")
    fasta = f"/home/delalamo/sabdab_structures/parsed/pdb_segments_complete/{fasta_chain}_clu_rep.fasta"
    seqnames = []
    for record in SeqIO.parse(fasta, "fasta"):
        seqnames.append(record.id)
    return seqnames

def load_embeddings(species, reps):
    topdir = f"/home/delalamo/sabr/npy_files_imgt/{species}/"
    rep_embed_dict = {}
    for rep in reps:
        rep_dir = os.path.join(topdir, rep + species[-2:])
        if not os.path.isdir(rep_dir):
            print(f"Directory {rep_dir} does not exist, skipping")
            continue
        rep_embed_dict[rep] = {}
        for npy_file in os.listdir(rep_dir):
            if not npy_file.endswith(".npy"):
                continue
            res = npy_file.split("_")[-1].split(".")[0]
            icode = res[-1] if not res[-1].isdigit() else ''
            res = res[:-1] if icode else res
            key = (res, icode)
            arr = np.load(os.path.join(rep_dir, npy_file))
            rep_embed_dict[rep][key] = arr
    return rep_embed_dict

def sort_arrays(array_dict, species, residue_selection):
    # Strategies need to include:
    # 1. All positions
    # 2. Only non-CDR positions
    # 3. Only non-CDR, non-insertion positions
    # 4. AHo-equivalent positions
    
    output = {}
    res_to_include = STRATEGIES[residue_selection]
    for rep, array_dict in array_dict.items():
        assert residue_selection in STRATEGIES.keys(), f"Residue selection {residue_selection} not recognized"
        
        for res in res_to_include:
            npy_file = f"/home/delalamo/sabr/npy_files_imgt/{species}/{rep}{species[-2:]}/imgt_res_{res}.npy"
            if not os.path.exists(npy_file):
                # print(f"File {npy_file} does not exist, skipping...")
                continue
            if res not in output:
                output[res] = np.zeros((0, 64))
            data = np.load(npy_file)
            output[res] = np.vstack([output[res], data])
    assert len(output) > 0, "No residues found for the given selection"
    out_array = np.zeros((0, 64))
    idxs = []
    for i, (idx, arr) in enumerate(output.items()):
        res_arr = np.mean(arr, axis=0)
        out_array = np.vstack([out_array, res_arr])
        idxs.append(idx)
    return out_array, idxs

def main():
    key = jax.random.PRNGKey(0)
    
    params_path = "/home/delalamo/SoftAlign/models/CONT_SW_05_T_3_1"
    params= pickle.load(open(params_path,"rb"))
    MODEL_ETE = hk.transform(model_end_to_end)

    # load all imgt npy arrays
    species_arrays = {}
    with open("/home/delalamo/sabr/strategies.txt", "w") as f:
        for strategy in STRATEGIES.keys():
            n_total_devs = 0
            n_cases = 0
            for species in ["camelid_H", "human_H", "human_L", "mouse_H", "mouse_L"]:
                if species not in species_arrays:
                    reps = get_reps(species)
                    rep_embed_dict = load_embeddings(species, reps)
                    print(f"Species {species} has {len(reps)} cluster representatives")
                    species_arrays[species] = rep_embed_dict
                else:
                    rep_embed_dict = species_arrays[species]
                for rep in reps:
                    try:
                        target_array, target_idxs = sort_arrays({rep: rep_embed_dict[rep]}, species, residue_selection = strategy)
                    except KeyError:
                        print(f"Skipping {rep} for {species} due to missing residues")
                        continue
                    ref_array, ref_idxs = sort_arrays({k: v for k, v in rep_embed_dict.items() if k != rep}, species, residue_selection = strategy)
                    # print(f"Target array shape", target_array.shape)
                    # print(f"Reference array shape", ref_array.shape)

                    # setup mpnn inputs
                    target_array = jnp.array(target_array[None, :])
                    ref_array = jnp.array(ref_array[None, :])
                    lens = jnp.array([target_array.shape[1], ref_array.shape[1]])[None,:]

                    soft_aln, sim_matrix, score = MODEL_ETE.apply(params,key, target_array, ref_array, lens, 10**-4)
                    devs = calc_devs(soft_aln, target_idxs, ref_idxs)
                    
                    if len(devs) > 0:
                        n_cases += 1
                        n_total_devs += len(devs)
                        print(f"\t{strategy} {species} {rep} vs rest: {score} {len(devs)} deviations: " + " ".join(f"{a}/{b}" for a,b in devs))
            f.write(f"{strategy},{n_cases},{n_total_devs}")
            print(f"{strategy},{n_cases},{n_total_devs}")


if __name__ == "__main__":
    main()