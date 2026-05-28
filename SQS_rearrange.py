from ase.io import read, write

# Define the desired order of elements
desired_order = ['Cs', 'Ca', 'Ge', 'Sn', 'Br']

# Load the structure from the POSCAR file
structure = read('C:/Data/Studies/PhD_Purdue/Projects/Water_splitting/Comps/POSCAR_sqs')

# Sort the atoms according to the desired order
sorted_structure = structure[[atom.index for element in desired_order for atom in structure if atom.symbol == element]]

# Write the sorted structure to a new POSCAR file
write('C:/Data/Studies/PhD_Purdue/Projects/Water_splitting/Comps/POSCAR_sorted', sorted_structure, format='vasp', direct=True)