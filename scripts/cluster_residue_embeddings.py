import os
import sys

sys.path.append('/home/delalamo/SoftAlign/')

import pickle

import END_TO_END_MODELS as ete
import haiku as hk
import Input_MPNN as input_
import jax
import jax.numpy as jnp
import MPNN as mpnn
import numpy as np
import Score_align as lddt
from Bio import PDB

# cycle through all PDBs
# make an numpy array that is 64 by 148 (or 149)
# for each PDB, encode
# assert encoding dimension matches # residues
# if so, go through residues and concatenate to numpy array
# set the rest to NAN

num_layers = 3 #@param {type:"integer"}
num_neighbors = 64 #@param {type:"integer"}
encoding_dim = 64 #@param {type:"integer"}
categorical = False #@param {type:"boolean"}
nb_clusters = 100 #@param {type:"integer"}
soft_max = False #@param {type:"boolean"}
bs = 15

def get_embeddings(
        x,
        mask,
        res,
        chain,
        node_features = encoding_dim,
        edge_features = encoding_dim,
        hidden_dim = encoding_dim,
        num_encoder_layers=num_layers,
        k_neighbors=num_neighbors,
        categorical = categorical,
        nb_clusters = nb_clusters,
        affine = True,
        soft_max = soft_max):
    a = ete.END_TO_END_SEQ_KMEANS(
            node_features,
            edge_features,
            hidden_dim,
            num_encoder_layers,
            k_neighbors,
            nb_clusters = nb_clusters,
            affine = True,
            soft_max = soft_max,
            dropout = 0.,
            augment_eps=0.0)
    return a.MPNN(x,mask,res,chain)


def model_end_to_end(
        x1,
        x2,
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
    if categorical:
        a = ete.END_TO_END_SEQ_KMEANS(
            node_features,
            edge_features,
            hidden_dim,
            num_encoder_layers,
            k_neighbors,
            nb_clusters = nb_clusters,
            affine = True,
            soft_max = soft_max,
            dropout = 0.,
            augment_eps=0.0)
    else:
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
    return a(x1,x2,lens,t)

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

def get_match_locs(arr):
    pass

def main():
    key = jax.random.PRNGKey(0)
    
    params_path = "/home/delalamo//SoftAlign/models/CONT_SW_05_T_3_1"
    params= pickle.load(open(params_path,"rb"))
    MODEL_ETE = hk.transform(model_end_to_end)

    GET_FN = hk.transform(get_embeddings)


    aho_fw_residues = list(range(1, 25)) + list(range(41, 58)) + list(range(78, 109)) + list(range(138, 150))
    print(aho_fw_residues)

    subdirs = ["human_H", "human_L", "mouse_H", "mouse_L"]
    for subdir in subdirs:

        topdir = f"/home/delalamo//sabdab_structures/parsed/aho/{subdir}/"

        aho_dicts = {x: np.full((0, 64), np.nan) for x in aho_fw_residues}

        print(topdir, len(os.listdir(topdir)))

        for file in os.listdir(topdir):
            file = os.path.join(topdir, file)
            if not file.endswith(".pdb"):
                continue
            
            print(file)
            X, mask, chain, res = input_.get_inputs_mpnn(file, chain = 'H')
            embeddings = GET_FN.apply(params, key, X, mask, res, chain)
            res_map = get_res_idx(file, 'H')
            
            print(embeddings.shape)
            print(len(res_map))

            if len(res_map) != embeddings.shape[1]:
                print(f"Error: {file} has {len(res_map)} residues but {embeddings.shape[1]} embeddings")
                continue

            for i, res in res_map.items():
                if res in aho_fw_residues:
                    new_arr = embeddings[0, i, :].reshape(1, -1)
                    aho_dicts[res] = np.concatenate((aho_dicts[res], new_arr), axis=0)

        for k, v in aho_dicts.items():
            np.save(f"npy_files/{subdir}/aho_res_{k}.npy", v)
    sys.exit()
    # then need to do this for all files, calculate the medoid, and save the results in some kind of format


if __name__ == "__main__":
    main()