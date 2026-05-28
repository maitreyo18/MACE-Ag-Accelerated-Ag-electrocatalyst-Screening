from icet import ClusterSpace, StructureContainer
from icet.tools.structure_generation import generate_sqs
from ase.io import read, write
from icet.tools.structure_generation import (generate_sqs,
                                             generate_sqs_from_supercells,
                                             generate_sqs_by_enumeration,
                                             generate_target_structure)
from ase import Atoms


# Load the initial structure from POSCAR
initial_structure = read('/Users/biswasm/Downloads/CONTCAR (4)')
print(initial_structure)

supercells = [initial_structure.repeat((2, 2, 2))]

chemical_symbols = chemical_symbols = [['Pb', 'Ba']]*2 + [['I']]*6 + [['N']]*4 + [['C']]*2 + [['H']]*10 
# Define the cluster space
cs = ClusterSpace(structure=initial_structure, cutoffs=[5], chemical_symbols=chemical_symbols)
print(cs)

target_comp = {'A': {'Pb': 0.50, 'Ba': 0.5}
               
    }

sqs = generate_sqs_from_supercells(cluster_space=cs,
                   supercells=supercells,
                   target_concentrations=target_comp)

desired_order = ['C', 'N', 'H', 'Ba', 'Pb', 'I']

reordered_sqs = Atoms(cell=sqs.cell, pbc=True)


for element in desired_order:
    for atom in sqs:
        if atom.symbol == element:
            reordered_sqs.append(atom)

file_path = '/Users/biswasm/Documents/Studies/PhD_Purdue/Projects/Water_splitting/DFT/Comps/mf1-comps/FABa_0.5Pb_0.5I_3-435/POSCAR_sqs_1'

# Write the SQS to a POSCAR file at the specified path
write(file_path, reordered_sqs, format='vasp', direct=True)

