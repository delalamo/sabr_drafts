import os

from Bio import PDB

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

parser = PDB.PDBParser(QUIET=True)

topdir = "/home/delalamo/sabdab_structures/parsed/pdb_segments_complete/"
for species in ["camelid_heavy", "human_heavy", "human_light", "mouse_heavy", "mouse_light"]:
    fasta = os.path.join(topdir, f"{species}.fasta")
    with open(fasta, "w") as f:
        for file in os.listdir(os.path.join(topdir, species)):
            if not file.endswith(".pdb"):
                continue
            filename = "_".join(file.split("_")[:2])
            struct = parser.get_structure("PDB", os.path.join(topdir, species, file))
            seq = ""
            chain = file.split("_")[1]
            for res in struct[0][chain]:
                if PDB.is_aa(res):
                    resname = res.get_resname()
                    resid = res.get_id()[1]
                    if (1 <= resid <= 128):
                        seq += AA_3TO1.get(resname, "")
            line = f">{filename}\n{seq}\n"
            print(line)
            f.write(line)
        