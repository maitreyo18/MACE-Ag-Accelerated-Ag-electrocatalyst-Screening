import pandas as pd
import numpy as np
import json
import logging
import warnings
import os
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import MACECalculator
import ase.io

# Torch
import torch
torch.set_default_dtype(torch.float64)

# Suppress warnings
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(
    filename='mace_testing_predictions_Ag_alloy.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Load MACE model
model_path = "./MACE_models_scratch/MACE_alloy+adsorb_stagetwo_compiled.model"
logger.info(f"Loading MACE model from: {model_path}")
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float64"
)
logger.info("MACE model loaded successfully!")

# File paths
train_val_xyz_path = '/depot/amannodi/data/Maitreyo_calcs/Ag-alloy_RPBE/Final_datasets_3_4/Combined_slab+O+OH+1_3_ML/train+val_MACE.xyz'
train_val_csv_path = '/depot/amannodi/data/Maitreyo_calcs/Ag-alloy_RPBE/Final_datasets_3_4/Combined_slab+O+OH+1_3_ML/train+val_final.csv'
test_xyz_path = '/depot/amannodi/data/Maitreyo_calcs/Ag-alloy_RPBE/Final_datasets_3_4/Combined_slab+O+OH+1_3_ML/test_MACE.xyz'
test_csv_path = '/depot/amannodi/data/Maitreyo_calcs/Ag-alloy_RPBE/Final_datasets_3_4/Combined_slab+O+OH+1_3_ML/test_final.csv'
valid_indices_path = './valid_indices_345.txt'
output_dir = './Train-Val-Test'
os.makedirs(output_dir, exist_ok=True)

# Load validation indices
logger.info("Loading validation indices...")
with open(valid_indices_path, 'r') as f:
    valid_indices = [int(line.strip()) for line in f.readlines()]
valid_indices_set = set(valid_indices)
logger.info(f"Loaded {len(valid_indices)} validation indices")

# Load datasets
logger.info("Loading train+val dataset from XYZ + CSV...")
atoms_list_train_val = ase.io.read(train_val_xyz_path, index=':')
df_train_val = pd.read_csv(train_val_csv_path)
assert len(atoms_list_train_val) == len(df_train_val)
logger.info(f"Loaded {len(atoms_list_train_val)} structures from train+val")

logger.info("Loading test dataset from XYZ + CSV...")
atoms_list_test = ase.io.read(test_xyz_path, index=':')
df_test = pd.read_csv(test_csv_path)
assert len(atoms_list_test) == len(df_test)
logger.info(f"Loaded {len(atoms_list_test)} structures from test")

# Predict with MACE
def predict_properties(atoms):
    atoms.calc = calc
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        return energy, forces, stress
    except Exception as e:
        logger.error(f"Error in MACE prediction: {e}")
        return None, None, None

# Convert arrays to lists
def array_to_list(arr):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr

# Process train+val
def process_train_val(atoms_list, df_meta, valid_indices_set):
    results = []
    for idx, (atoms, row) in enumerate(zip(atoms_list, df_meta.itertuples())):
        is_validation = idx in valid_indices_set
        structure_name = row.Alloy_name
        structure_json = row.Structure
        actual_energy = row.Energy
        actual_forces = json.loads(row.Forces) if isinstance(row.Forces, str) else row.Forces
        actual_stress = json.loads(row.Stresses) if isinstance(row.Stresses, str) else row.Stresses

        pred_energy, pred_forces, pred_stress = predict_properties(atoms)
        if pred_energy is not None:
            logger.info(f"{structure_name}: Actual E = {actual_energy:.4f}, Pred E = {pred_energy:.4f}")

        results.append({
            'Structure Name': structure_name,
            'Structure': structure_json,
            'Actual Energy': actual_energy,
            'Actual Forces': actual_forces,
            'Actual Stress': actual_stress,
            'Pred Energy': pred_energy,
            'Pred Forces': array_to_list(pred_forces),
            'Pred Stress': array_to_list(pred_stress),
            'Is Validation': is_validation
        })

    train_results = [r for r in results if not r['Is Validation']]
    val_results = [r for r in results if r['Is Validation']]
    for r in train_results + val_results:
        del r['Is Validation']

    pd.DataFrame(train_results).to_csv(os.path.join(output_dir, "train_set.csv"), index=False)
    pd.DataFrame(val_results).to_csv(os.path.join(output_dir, "val_set.csv"), index=False)
    logger.info(f"Saved train_set.csv ({len(train_results)}), val_set.csv ({len(val_results)})")
    return train_results, val_results

# Process test
def process_test(atoms_list, df_meta):
    results = []
    for atoms, row in zip(atoms_list, df_meta.itertuples()):
        structure_name = row.Alloy_name
        structure_json = row.Structure
        actual_energy = row.Energy
        actual_forces = json.loads(row.Forces) if isinstance(row.Forces, str) else row.Forces
        actual_stress = json.loads(row.Stresses) if isinstance(row.Stresses, str) else row.Stresses

        pred_energy, pred_forces, pred_stress = predict_properties(atoms)
        if pred_energy is not None:
            logger.info(f"{structure_name}: Actual E = {actual_energy:.4f}, Pred E = {pred_energy:.4f}")

        results.append({
            'Structure Name': structure_name,
            'Structure': structure_json,
            'Actual Energy': actual_energy,
            'Actual Forces': actual_forces,
            'Actual Stress': actual_stress,
            'Pred Energy': pred_energy,
            'Pred Forces': array_to_list(pred_forces),
            'Pred Stress': array_to_list(pred_stress)
        })

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "test_set.csv"), index=False)
    logger.info(f"Saved test_set.csv ({len(results)})")
    return results

# --- Main ---
print("Starting MACE prediction on train+val dataset...")
train_results, val_results = process_train_val(atoms_list_train_val, df_train_val, valid_indices_set)

print("Starting MACE prediction on test dataset...")
test_results = process_test(atoms_list_test, df_test)

print("All MACE processing complete! Check './Train-Val-Test' for results.")
logger.info("All processing done.")
print(f"\nSummary:\nTrain: {len(train_results)}, Val: {len(val_results)}, Test: {len(test_results)}")

