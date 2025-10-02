import sys

sys.path.append("/home/delalamo/SoftAlign/")

import os

import END_TO_END_MODELS as ete
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import PDB, SeqIO

num_layers = 3  # @param {type:"integer"}
num_neighbors = 64  # @param {type:"integer"}
encoding_dim = 64  # @param {type:"integer"}
categorical = False  # @param {type:"boolean"}
nb_clusters = 100  # @param {type:"integer"}
soft_max = False  # @param {type:"boolean"}
bs = 15

ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STRATEGIES = {
    "NO_INS": [f"{x}" for x in range(1, 129)],
    # "ALL": [f"{x}{y}" for x in range(1, 129) for y in ALPHABET],
    # "NO_CDR": [
    #     f"{x}{y}"
    #     for x in list(range(1, 26))
    #     + list(range(39, 56))
    #     + list(range(66, 105))
    #     + list(range(118, 129))
    #     for y in ALPHABET
    # ],
    # "NO_CDR_NO_INS": [
    #     f"{x}"
    #     for x in list(range(1, 26))
    #     + list(range(39, 56))
    #     + list(range(66, 105))
    #     + list(range(118, 129))
    # ],
    # "AHO": [str(x) for x in list(range(1, 33))]
    # + ["33A", "33B"]
    # + [str(x) for x in list(range(34, 120))]
    # + [f"120{x}" for x in ALPHABET]
    # + [str(x) for x in list(range(121, 129))],
}

STRATEGIES = {k: [x.strip() for x in v] for k, v in STRATEGIES.items()}


def model_end_to_end(
    targ_arr,
    ref_arr,
    lens,
    t,
    node_features=encoding_dim,
    edge_features=encoding_dim,
    hidden_dim=encoding_dim,
    num_encoder_layers=num_layers,
    k_neighbors=num_neighbors,
    categorical=categorical,
    nb_clusters=nb_clusters,
    affine=True,
    soft_max=soft_max,
):
    a = ete.END_TO_END(
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers,
        k_neighbors,
        affine=True,
        soft_max=soft_max,
        dropout=0.0,
        augment_eps=0.0,
    )

    return a.align(targ_arr, ref_arr, lens, t)


def get_res_idx(pdb, chain):
    """
    Get the residue index for a given chain in a PDB file.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb)
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
    fasta = (
        f"/home/delalamo/sabdab_structures/parsed/"
        f"pdb_segments_complete/{fasta_chain}_clu_rep.fasta"
    )
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
        rep_dict = {}
        try:
            for npy_file in os.listdir(rep_dir):
                if not npy_file.endswith(".npy"):
                    continue
                res = npy_file.split("_")[-1].split(".")[0]
                icode = res[-1] if not res[-1].isdigit() else ""
                res = res[:-1] if icode else res
                key = (res, icode)
                arr = np.load(os.path.join(rep_dir, npy_file))
                rep_dict[key] = arr
        except EOFError:
            print(f"Error loading embeddings for {rep}, skipping")
            continue
        rep_embed_dict[rep] = rep_dict
        # print(rep, len(rep_embed_dict[rep]))
    return rep_embed_dict


def sort_arrays(array_dict, species, residue_selection):
    # Strategies need to include:
    # 1. All positions
    # 2. Only non-CDR positions
    # 3. Only non-CDR, non-insertion positions
    # 4. AHo-equivalent positions

    output = {}
    res_to_include = STRATEGIES[residue_selection]
    for rep in array_dict.keys():
        assert (
            residue_selection in STRATEGIES.keys()
        ), f"Residue selection {residue_selection} not recognized"

        for res in res_to_include:
            npy_file = (
                f"/home/delalamo/sabr/npy_files_imgt/"
                f"{species}/{rep}{species[-2:]}/imgt_res_{res}.npy"
            )
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
    for idx, arr in output.items():
        res_arr = np.mean(arr, axis=0)
        out_array = np.vstack([out_array, res_arr])
        idxs.append(idx)
    return out_array, idxs


def read_cluster_definitions(species):
    recorded_entries = [
        x[:6]
        for x in os.listdir((f"/home/delalamo/sabr/npy_files_imgt/" f"{species}/"))
    ]

    clu_name = species.replace("_H", "_heavy").replace("_L", "_light")
    clufile = (
        f"/home/delalamo/sabdab_structures/parsed/"
        f"pdb_segments_complete/{clu_name}.tsv"
    )
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
    lowest_resolutions = []
    all_resolutions = []

    for cluster, members in species_clusters.items():
        members4 = [m.lower()[:4] for m in members]
        cluster_df = df[df["pdb"].isin(members4)]
        all_resolutions.extend(cluster_df["resolution"].dropna().tolist())
        if cluster_df.empty:
            print(
                (
                    f"No entries found in summary for cluster {cluster} "
                    f"of size {len(members)}, skipping..."
                )
            )
            continue
        # Drop rows with NaN in 'resolution' before finding the minimum
        df_clean = cluster_df.dropna(subset=["resolution"])
        if df_clean.empty:
            print(f"All entries for cluster {cluster} have NaN resolution")
            continue
        highest_res_pdb = df_clean.loc[df_clean["resolution"].idxmin()]["pdb"]
        lowest_res = df_clean.loc[df_clean["resolution"].idxmin()]["resolution"]
        lowest_resolutions.append(lowest_res)
        for m in members:
            if m.lower().startswith(highest_res_pdb.lower()):
                reps.append(m)
                break

    def safe_float(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    all_resolutions = [safe_float(r) for r in all_resolutions]
    all_resolutions = [r for r in all_resolutions if r is not None]

    lowest_resolutions = [safe_float(r) for r in lowest_resolutions]
    lowest_resolutions = [r for r in lowest_resolutions if r is not None]
    return reps, lowest_resolutions, all_resolutions


def plot_resolution_data(species, lowest_resolutions, all_resolutions, max_res=7.5):
    lowest_resolutions = [r for r in lowest_resolutions if r <= max_res]
    all_resolutions = [r for r in all_resolutions if r <= max_res]
    plt.figure(figsize=(10, 6))
    plt.xlim(0, max_res)
    plt.hist(
        all_resolutions, bins=30, alpha=0.5, label="SAbDaB", color="blue", density=True
    )
    plt.hist(
        lowest_resolutions,
        bins=30,
        alpha=0.7,
        label="SAbR representatives",
        color="orange",
        density=True,
    )

    sns.kdeplot(all_resolutions, color="blue")
    sns.kdeplot(lowest_resolutions, color="orange")
    plt.xlabel("Resolution (Å)")
    plt.ylabel("Frequency")
    plt.title(f"{species}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        (
            f"/home/delalamo/sabr/resolution_plots/"
            f"{species}_resolution_distribution.png"
        )
    )
    plt.close()


def main():

    all_species = ["camelid_H", "human_H", "human_L", "mouse_H", "mouse_L"]

    # load all imgt npy arrays
    species_arrays = {}

    # need to load tsv and cluster definitions, fetch highest-resolution one
    # reps = get_reps(species)
    df = pd.read_csv(
        "/home/delalamo/sabdab_structures/sabdab_summary_all.tsv",
        sep="\t",
        engine="python",
        on_bad_lines="warn",
    )

    cluster_assignments = {s: read_cluster_definitions(s) for s in all_species}
    rep_dict = {}
    rep_resolution_dict = {}
    resolution_dict = {}
    for s in all_species:
        reps, lowest_resolutions, all_resolutions = fetch_highest_res_reps(
            cluster_assignments[s], df
        )
        rep_dict[s] = reps
        rep_resolution_dict[s] = lowest_resolutions
        resolution_dict[s] = all_resolutions
        print(s, len(rep_resolution_dict[s]))
        plot_resolution_data(s, lowest_resolutions, all_resolutions)

    all_data = {}
    for species in all_species:
        rep_embed_dict = load_embeddings(
            species, [m for v in cluster_assignments[species].values() for m in v]
        )
        species_arrays[species] = rep_embed_dict
        # reps = [r for r in rep_dict[species]]
        reps = [r for r in rep_dict[species] if r in rep_embed_dict.keys()]
        for strategy in STRATEGIES.keys():
            ref_array, ref_idxs = sort_arrays(
                {r: rep_embed_dict[r] for r in reps},
                species,
                residue_selection=strategy,
            )
            all_data[species] = {"array": ref_array, "idxs": ref_idxs}
            print(f"{species} [{strategy}]: {ref_array.shape}, {len(ref_idxs)}")
    np.savez_compressed("/home/delalamo/sabr/ref_arrays/embeddings.npz", all_data)


if __name__ == "__main__":
    main()
