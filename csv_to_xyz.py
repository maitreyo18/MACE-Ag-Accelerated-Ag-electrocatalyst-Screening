import os
import json
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from ase import Atoms

def pymatgen_df_to_ase_atoms(df):
    """
    Convert a DataFrame with pymatgen structure dict, stresses, forces, and energy
    to a list of ASE Atoms objects with forces, stresses, and energy included.
    Args:
    df (pd.DataFrame): DataFrame with columns 'Structure', 'Stresses', 'Forces', 'Energy', 'Directory'
    Returns:
    list: List of ASE Atoms objects
    """
    atoms_list = []
    for _, row in df.iterrows():
        # Convert pymatgen Structure dict (stored as JSON string) to Structure object
        pmg_structure = Structure.from_dict(json.loads(row['Structure']))
        # Create ASE Atoms object
        atoms = Atoms(
            symbols=[site.specie.symbol for site in pmg_structure],
            positions=[site.coords for site in pmg_structure],
            cell=pmg_structure.lattice.matrix,
            pbc=True
        )
        # Add energy (assuming it's in eV)
        atoms.info['energy_pbe'] = row['Energy']
        # Add forces (assuming they're in eV/Angstrom)
        atoms.arrays['forces_pbe'] = np.array(json.loads(row['Forces']))
        # Add stresses (3x3 matrix in eV/Angstrom^3) — keep full 9 components
        stress = np.array(json.loads(row['Stresses']))
        atoms.info['stress_pbe'] = stress.flatten()
        # Add source information
        atoms.info['source'] = row['Directory']
        atoms_list.append(atoms)
    return atoms_list

def create_single_xyz(atoms_list, output_file='train+val_MACE.xyz'):
    """
    Create a single XYZ file containing all structures from the atoms_list,
    with a specific comment line format including lattice, properties, energy, and stress.
    Args:
    atoms_list (list): List of ASE Atoms objects
    output_file (str): Path of the single XYZ file to create
    Returns:
    str: Path of the created XYZ file
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w') as f:
        for atoms in atoms_list:
            f.write(f"{len(atoms)}\n")
            # Prepare comment line components
            lattice = atoms.get_cell()
            lattice_str = f"{lattice[0][0]} {lattice[0][1]} {lattice[0][2]} {lattice[1][0]} {lattice[1][1]} {lattice[1][2]} {lattice[2][0]} {lattice[2][1]} {lattice[2][2]}"
            properties = "species:S:1:pos:R:3:forces_pbe:R:3"
            energy = atoms.info.get('energy_pbe')
            stress = atoms.info.get('stress_pbe')
            # Always expand stress as 9 components
            stress_str = f"{stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]} {stress[6]} {stress[7]} {stress[8]}"
            # Write the formatted comment line
            comment = f'Lattice="{lattice_str}" Properties={properties} energy_pbe={energy:.6f} stress_pbe="{stress_str}" pbc="T T T"'
            f.write(comment + "\n")
            # Write atomic data
            if atoms.has('forces_pbe'):
                forces = atoms.get_array('forces_pbe')
                for atom, force in zip(atoms, forces):
                    symbol = atom.symbol
                    position = atom.position
                    f.write(f"{symbol} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} {force[0]:.6f} {force[1]:.6f} {force[2]:.6f}\n")
            else:
                for atom in atoms:
                    symbol = atom.symbol
                    position = atom.position
                    f.write(f"{symbol} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} 0.000000 0.000000 0.000000\n")
    return output_file

if __name__ == "__main__":
    # Read the updated CSV
    df = pd.read_csv("train+val_final.csv")
    atoms_list = pymatgen_df_to_ase_atoms(df)
    create_single_xyz(atoms_list, output_file="train+val_MACE.xyz")

