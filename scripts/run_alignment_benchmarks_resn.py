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
import pandas as pd
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

def read_cluster_definitions(species):
    recorded_entries = [x[:6] for x in os.listdir(f"/home/delalamo/sabr/npy_files_imgt/{species}/")]
    
    clu_name = species.replace("_H", "_heavy").replace("_L", "_light")
    clufile = f"/home/delalamo/sabdab_structures/parsed/pdb_segments_complete/{clu_name}.tsv"
    clusters = {}
    with open(clufile) as f:
        for line in f:
            if line.startswith("#") or len(line.split()) < 2:
                continue
            cluster_name, pdb_id = line.strip().split()
            if pdb_id not in recorded_entries:
                continue
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(pdb_id)
    return clusters

def fetch_highest_res_reps(species_clusters, df):
    reps = []
    
    for cluster, members in species_clusters.items():
        members4 = [m.lower()[:4] for m in members]
        cluster_df = df[df['pdb'].isin(members4)]
        if cluster_df.empty:
            print(f"No entries found in summary for cluster {cluster} of size {len(members)}, skipping...")
            continue
        # Drop rows with NaN in 'resolution' before finding the minimum
        cluster_df_clean = cluster_df.dropna(subset=['resolution'])
        if cluster_df_clean.empty:
            print(f"All entries for cluster {cluster} have NaN resolution, skipping...")
            continue        
        highest_res_pdb = cluster_df_clean.loc[cluster_df_clean['resolution'].idxmin()]["pdb"]
        for m in members:
            if m.lower().startswith(highest_res_pdb.lower()):
                reps.append(m)
                break
    return reps

def main():
    key = jax.random.PRNGKey(0)
    
    all_species = ["camelid_H", "human_H", "human_L", "mouse_H", "mouse_L"]

    params_path = "/home/delalamo/SoftAlign/models/CONT_SW_05_T_3_1"
    params= pickle.load(open(params_path,"rb"))
    MODEL_ETE = hk.transform(model_end_to_end)

    # load all imgt npy arrays
    species_arrays = {}

    cdr_residues = list(range(26, 39)) + list(range(56, 66)) + list(range(105, 118))

    # need to load tsv and cluster definitions, fetch highest-resolution one
    # reps = get_reps(species)
    df = pd.read_csv("/home/delalamo/sabdab_structures/sabdab_summary_all.tsv", sep='\t', engine='python', on_bad_lines='warn')

    cluster_assignments = {s: read_cluster_definitions(s) for s in all_species}
    rep_dict = {s: fetch_highest_res_reps(cluster_assignments[s], df) for s in all_species}
    for species in all_species:
        rep_embed_dict = load_embeddings(species, [m for v in cluster_assignments[species].values() for m in v])
        species_arrays[species] = rep_embed_dict

    outfile = "/home/delalamo/sabr/strategies_resn.txt"
    if os.path.exists(outfile):
        out_df = pd.read_csv(outfile)
    else:
        out_df = pd.DataFrame(columns=["strategy", "species", "pdb", "n_devs", "n_non_cdr_devs", "devs"])
        out_df.to_csv(outfile, index=False)
    try:
        for strategy in STRATEGIES.keys():
            n_total_devs = 0
            n_cases = 0
            for species in all_species:
                rep_embed_dict = species_arrays[species]
                # need to cycle through all clusters

                for rep, members in cluster_assignments[species].items():
                    other_reps = [r for r in rep_dict[species] if r not in members]
                    ref_array, ref_idxs = sort_arrays({r: rep_embed_dict[r] for r in other_reps}, species, residue_selection = strategy)
                    ref_array = jnp.array(ref_array[None, :])
                    for member in members:

                        # Check if this row already exists in df and skip if true
                        if ((out_df["strategy"] == strategy) & (out_df["species"] == species) & (out_df["pdb"] == member)).any():
                            print(f"\tCalculations for {member} ({species}) already executed for strategy {strategy}, skipping")
                            continue
                        try:
                            target_array, target_idxs = sort_arrays({member: rep_embed_dict[member]}, species, residue_selection = strategy)
                        except KeyError:
                            print(f"\tSkipping {rep} for {species} due to missing residues")
                            continue

                        # setup mpnn inputs
                        target_array = jnp.array(target_array[None, :])
                        
                        # print(species, rep, member, target_array.shape, ref_array.shape)
                        lens = jnp.array([target_array.shape[1], ref_array.shape[1]])[None,:]

                        soft_aln, sim_matrix, score = MODEL_ETE.apply(params,key, target_array, ref_array, lens, 10**-4)
                        devs = calc_devs(soft_aln, target_idxs, ref_idxs)
                        non_cdr_devs = [d for d in devs if d[0] not in cdr_residues]
                        
                        # get resolution
                        resolution = 0.0
                        member_pdb = member[:4].lower()
                        row = df[df['pdb'].str.lower() == member_pdb]
                        if not row.empty and 'resolution' in row.columns:
                            resolution = row.iloc[0]['resolution']

                        out_df = pd.concat([out_df, pd.DataFrame([{
                            "strategy": strategy,
                            "species": species,
                            "pdb": rep,
                            "n_devs": len(devs),
                            "n_non_cdr_devs": len(non_cdr_devs),
                            "devs": " ".join(f"{a}/{b}" for a, b in devs)
                        }])], ignore_index=True)
                        out_df.to_csv(outfile, index=False)
                        if len(non_cdr_devs) > 0:
                            n_cases += 1
                            n_total_devs += len(non_cdr_devs)
                            print(f"\t{strategy} {species} {member} vs rest: {score} {len(devs)} deviations ({len(non_cdr_devs)} non-CDRs; {resolution} A): " + " ".join(f"{a}/{b}" for a,b in devs))
                        else:
                            print(f"\t{strategy} {species} {member} vs rest: {score} 0 deviations ({resolution} A)")
                        out_df.to_csv(outfile, index=False)
                #f.write(f"{strategy},{n_cases},{n_total_devs}")
                print(f"{strategy},{n_cases},{n_total_devs}")
    except KeyboardInterrupt:
        out_df.to_csv(outfile, index=False)
        print("Interrupted, exiting...")


if __name__ == "__main__":
    main()