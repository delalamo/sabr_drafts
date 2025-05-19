import sys

sys.path.append('/home/delalamo/SoftAlign/')

import pickle

import END_TO_END_MODELS as ete
import haiku as hk
import Input_MPNN as input_
import jax
import jax.numpy as jnp
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

    pdb1 = "/home/delalamo//sabdab_structures/parsed/aho/camelid_H/1bzq_K_H_0001.pdb"
    pdb2 = "/home/delalamo//sabdab_structures/parsed/aho/camelid_H/9ljj_N_H_0001.pdb"
    X2,mask2,chain2,res2 = input_.get_inputs_mpnn(pdb2, chain = 'H')
    X1,mask1,chain1,res1 = input_.get_inputs_mpnn(pdb1, chain = 'H')
    MODEL_ETE = hk.transform(model_end_to_end)

    x1 = X1,mask1,chain1,res1
    x2 = X2,mask2,chain2,res2
    lens = jnp.array([X1.shape[1],X2.shape[1]])[None,:]

    soft_aln,sim_matrix,score = MODEL_ETE.apply(params,key, x1,x2,lens,10**-4)

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