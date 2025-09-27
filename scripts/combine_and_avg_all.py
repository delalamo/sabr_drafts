import os

import numpy as np


def main():
    # need to load all npy files for camelid, human_H, human_L and stack them 
    topdir = "/home/delalamo/sabr/npy_files_imgt"
    for organism in ["camelid_H", "human_H", "human_L"]:
        organism_dir = os.path.join(topdir, organism)
        arr = np.empty((0, 64))
        for filename in os.listdir(organism_dir):
            residue = filename.split("_")[-1].split(".")[0]
            if not filename.endswith(".npy"):
                continue
            data = np.load(os.path.join(organism_dir, filename))
            n_seqs = data.shape[0]
            avg = np.mean(data, axis=0)
            arr = np.vstack([data, arr])
            print(f"Adding residue {residue} ({n_seqs})")
        print(f"Final shape for {organism}: {arr.shape}")
        np.save(os.path.join(topdir, f"{organism}_all.npy"), arr)
    pass
    

if __name__ == "__main__":
    main()