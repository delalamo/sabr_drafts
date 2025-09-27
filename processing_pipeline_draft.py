import os

"""
overall processing pipeline for sabdab
1) download PDB files from sabdab, imgt format
2) open tsv summary file
3) parse PDB files and extract chains, split by species and heavy/light
4) also write chains res 1-128 to fasta files 
5) cluster with mmseqs2
6) for each cluster, select representative (best resolution?)
7) embed these structures
8) average and stack per-residue embeddings 
"""