import os
import sys

from Bio import PDB

PARSER = PDB.PDBParser(QUIET=True)

def extract_and_save_chain_by_filename(pdb_file_path, output_dir):
    """
    Extracts a chain specified by the input filename's 6th character
    and saves it to a new PDB file.
    """
    filename = os.path.basename(pdb_file_path)

    target_chain_id = filename[5]  # 6th character is the chain ID
    base, ext = os.path.splitext(filename)
    output_filename = f"{base[:4]}_{target_chain_id}_{output_dir[-1].upper()}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        print(f"File '{output_path}' already exists. Skipping extraction.")
        return

    # Ensure filename is long enough and is a .pdb file
    if not filename.endswith(".pdb"):
        print(f"Skipping '{filename}': Invalid filename format or not a PDB file.")
        return



    try:
        original_structure = PARSER.get_structure("original_structure", pdb_file_path)
    except Exception as e:
        print(f"Error parsing PDB file '{pdb_file_path}': {e}")
        return

    extracted_chain_object = None
    # Iterate through models to find the chain
    for model in original_structure:
        if target_chain_id in model:
            chain_to_extract = model[target_chain_id]
            chain_to_extract.detach_parent()  # Detach from the original model
            extracted_chain_object = chain_to_extract
            chain_to_extract.id = output_dir[-1].upper()  # Rename the chain ID to 'A'
            break  # Assuming the first model containing the chain is sufficient
    
    if extracted_chain_object:
        # Create a new Structure and Model to hold the extracted chain
        new_structure = PDB.Structure.Structure("extracted_s") # New structure ID
        new_model = PDB.Model.Model(0)                     # New model ID (typically 0)
        
        new_model.add(extracted_chain_object) # Add the (detached) chain to the new model
        new_structure.add(new_model)          # Add the new model to the new structure
        
        io = PDB.PDBIO()
        io.set_structure(new_structure)
        
        # Construct output filename
        
        try:
            io.save(output_path)
            print(f"Successfully extracted chain '{target_chain_id}' from '{filename}' to '{output_path}'")
        except Exception as e:
            print(f"Error saving extracted chain to '{output_path}': {e}")
    else:
        print(f"Chain '{target_chain_id}' (from filename) not found in '{filename}'.")

if __name__ == "__main__":
    
    
    input_dirs = [
        "/home/delalamo/sabdab_structures/parsed/pdb_segments/human_heavy",
        "/home/delalamo/sabdab_structures/parsed/pdb_segments/human_light",
        "/home/delalamo/sabdab_structures/parsed/pdb_segments/mouse_heavy",
        "/home/delalamo/sabdab_structures/parsed/pdb_segments/mouse_light"
    ]
    output_dirs = [
        "/home/delalamo/sabdab_structures/parsed/fixed_chains/human_H",
        "/home/delalamo/sabdab_structures/parsed/fixed_chains/human_L",
        "/home/delalamo/sabdab_structures/parsed/fixed_chains/mouse_H",
        "/home/delalamo/sabdab_structures/parsed/fixed_chains/mouse_L"
    ]
    
    for input_directory, output_directory in zip(input_dirs, output_dirs):
        # Create the output directory if it doesn't exist
        for item in os.listdir(input_directory):
            if item.lower().endswith(".pdb"):
                full_path = os.path.join(input_directory, item)
                if os.path.isfile(full_path): # Ensure it's a file
                    extract_and_save_chain_by_filename(full_path, output_directory)