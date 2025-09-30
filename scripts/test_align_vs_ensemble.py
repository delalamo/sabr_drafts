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
from Bio import PDB

num_layers = 3 #@param {type:"integer"}
num_neighbors = 64 #@param {type:"integer"}
encoding_dim = 64 #@param {type:"integer"}
categorical = False #@param {type:"boolean"}
soft_max = False #@param {type:"boolean"}

def model_end_to_end(
        x1,
        ref_arr,
        lens,
        t,
        node_features = encoding_dim,
        edge_features = encoding_dim,
        hidden_dim = encoding_dim,
        num_encoder_layers=num_layers,
        k_neighbors=num_neighbors,
        affine = True,
        soft_max = soft_max
    ):
    a = ete.END_TO_END(
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers,
        k_neighbors,
        affine = affine,
        soft_max = soft_max,
        dropout = 0.,
        augment_eps=0.0)
    X1,mask1,res1,ch1 = x1
    targ_arr = a.MPNN(X1,mask1,res1,ch1)

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

def sort_arrays(species_arrays, include_cdrs=False, include_insertions=False):
    output = {}
    for species, array_dict in species_arrays.items():
        alphabet = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        new_array = np.zeros((0, 64))
        res_idxs = []
        res_to_include = list(range(1, 26)) + list(range(39, 56)) + list(range(66, 105)) + list(range(118, 129))
        if include_cdrs:
            res_to_include = range(1, 129)
        for i in res_to_include:
            if include_insertions:
                for icode in alphabet:
                    key = f"{i}{icode if icode != ' ' else ''}"
                    if key in array_dict:
                        new_array = np.vstack([new_array, array_dict[key]])
                        res_idxs.append(key)
            else:
                key = str(i)
                if key in array_dict:
                    new_array = np.vstack([new_array, array_dict[key]])
                    res_idxs.append(key)
        output[species] = (new_array, res_idxs)
    return output

def calc_devs(soft_aln, targ_idxs, ref_idxs):
    devs = []
    coords = np.argwhere(np.array(soft_aln[0]) == 1)
    for i, j in coords:
        res_target = int(targ_idxs[i].split(".")[-1])
        res_ref = int(ref_idxs[j])
        if res_target != res_ref:
            devs.append((res_target, res_ref))
    return devs

def main():
    key = jax.random.PRNGKey(0)
    
    params_path = "/home/delalamo/SoftAlign/models/CONT_SW_05_T_3_1"
    params= pickle.load(open(params_path,"rb"))

    # load all imgt npy arrays
    species_arrays = {}
    for species in ["camelid_H", "human_H", "human_L", "mouse_H", "mouse_L"]:
        topdir = f"/home/delalamo/sabr/npy_files_imgt/{species}/"
        species_arrays[species] = {}
        for file in os.listdir(topdir):
            if not file.endswith(".npy"):
                continue
            res = file.split("_")[-1].split(".")[0]
            arr = np.load(os.path.join(topdir, file))
            species_arrays[species][res] = np.mean(arr, axis=0)
            assert species_arrays[species][res].shape[-1] == 64
    
    species_arrays = sort_arrays(species_arrays)
    MODEL_ETE = hk.transform(model_end_to_end)

    pdb1 = "/home/delalamo/sabdab_structures/parsed/fixed_chains/camelid_H/1bzq_K_H.pdb"
    X1,mask1,chain1,res1, ids = input_.get_inputs_mpnn(pdb1, chain = 'H')
    x1 = X1,mask1,chain1,res1
    
    for species, (array, res_idxs) in species_arrays.items():
        
        to_run = jnp.array(array[None, :])
        lens = jnp.array([X1.shape[1], to_run.shape[1]])[None,:]

        soft_aln, sim_matrix, score = MODEL_ETE.apply(params,key, x1, to_run, lens, 10**-4)
        devs = calc_devs(soft_aln, ids, res_idxs)
        print(species, score, len(devs), " ".join(f"{a}/{b}" for a,b in devs))
        continue

        mask__1 = np.ones((1,X1.shape[1],X1.shape[1]))
        mask__2 = np.ones((1,X2.shape[1],X2.shape[1]))

        print(lddt.get_LDDTloss(X1[:,:,1],X2[:,:,1],soft_aln,mask__1,mask__2,10**-4))
        print(lddt.get_LDDTloss(X2[:,:,1],X1[:,:,1],soft_aln.transpose(0,2,1),mask__2,mask__1,10**-4))

        print(soft_aln.shape)
        idxs = np.argwhere(soft_aln[0] > 0.5)

        res_idx_A = get_res_idx(pdb1, 'H')
        res_idx_B = get_res_idx(pdb2, 'H')

        for i in range(idxs.shape[0]):
            res_A = res_idx_A[idxs[i,0]]
            res_B = res_idx_B[idxs[i,1]]
            print(f"Residue {res_A} in PDB1 matches with residue {res_B} in PDB2 with score {soft_aln[0][idxs[i,0]][idxs[i,1]]}")



if __name__ == "__main__":
    main()