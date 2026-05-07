#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an M3GNet potential using MatGL + PyTorch Lightning.

Key features:
- Default Adam optimizer
- CosineAnnealingLR schedule (lr=1e-3 → eta_min=5e-4 over 500 epochs)
- Huber loss with delta=0.01
- Manual elemental reference energies (embedded into DEFAULT_ELEMENTS order)
"""

from __future__ import annotations

import os
import warnings
import logging
import json
import ast
import numpy as np
import pandas as pd
import dgl
import torch
import pytorch_lightning as pl
from functools import partial
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pytorch_lightning.loggers import CSVLogger

# MatGL components
import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.layers import AtomRef
from matgl.config import DEFAULT_ELEMENTS

# Your dataset utilities
from mgl_data_utils_1 import MGLDataset, MGLDataLoader, collate_fn_pes


# ===============================================================
#  ENVIRONMENT & LOGGING
# ===============================================================
os.environ["LD_LIBRARY_PATH"] = (
    "/apps/gilbreth/cuda-toolkit/cuda-11.2.0/lib64:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

print("Using DGL version:", dgl.__version__)

log_file = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("=== M3GNet Training Script Started ===")


# ===============================================================
#  TORCH CONFIGURATION
# ===============================================================
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("medium")
warnings.simplefilter("ignore")

logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")


# ===============================================================
#  LOAD DATASET
# ===============================================================
csv_path = "./train+val_final.csv"
df = pd.read_csv(csv_path)
logger.info(f"Loaded dataset: {csv_path} with {len(df)} entries")

# Parse numeric arrays
df["Forces_array"] = df["Forces"].apply(lambda x: np.array(ast.literal_eval(x)))
df["Stresses_array"] = df["Stresses"].apply(
    lambda x: np.array(ast.literal_eval(x)) * -0.1
)

# Extract data
structures = [Structure.from_dict(json.loads(s)) for s in df["Structure"]]
structure_names = df["Alloy_name"].tolist()
energies = df["Energy"].astype(float).tolist()
forces = [f.tolist() for f in df["Forces_array"]]
stresses = [s.tolist() for s in df["Stresses_array"]]

labels = {"energies": energies, "forces": forces, "stresses": stresses}
logger.info(f"Parsed {len(structures)} structures with energy/force/stress labels")


# ===============================================================
#  DETECT ELEMENTS USED IN DATASET
# ===============================================================
dataset_elements = sorted({el.symbol for st in structures for el in st.composition.elements})
logger.info(f"Detected elements in dataset: {dataset_elements}")


# ===============================================================
#  CONVERTER & DATASET
# ===============================================================
element_types = DEFAULT_ELEMENTS  # full list (89 elements)
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

save_dir = "./mgl_dataset"
os.makedirs(save_dir, exist_ok=True)

dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    structure_names=structure_names,
    converter=converter,
    labels=labels,
    include_line_graph=True,
    save_dir=save_dir,
)

# Split into train/val
train_data, val_data, test_data = dgl.data.utils.split_dataset(
    dataset,
    frac_list=[0.9, 0.1, 0.0],
    shuffle=True,
    random_state=79,
)

# Save split info
pd.DataFrame(
    [structure_names[i] for i in train_data.indices], columns=["Structure Name"]
).to_csv(os.path.join(save_dir, "train_names.csv"), index=False)
pd.DataFrame(
    [structure_names[i] for i in val_data.indices], columns=["Structure Name"]
).to_csv(os.path.join(save_dir, "val_names.csv"), index=False)
logger.info("Saved train/validation structure name lists.")


# ===============================================================
#  DATA LOADERS
# ===============================================================
my_collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=True)
generator = torch.Generator(device="cuda").manual_seed(42)

train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=16,
    num_workers=0,
    generator=generator,
)
logger.info("Initialized DGL DataLoaders.")


# ===============================================================
#  MANUAL ELEMENTAL REFERENCES
# ===============================================================
logger.info("Using manual elemental reference energies embedded in DEFAULT_ELEMENTS order...")

# Manual reference energies (eV/atom) for dataset elements
manual_refs = {
    "Ag": -2.185981,
    "Au": -2.623971,
    "Bi": -3.446913,
    "Cu": -3.248735,
    "O": -1.758624,
    "H": -3.492632,
    "Pd": -4.629983,
    "Sn": -3.320969
}

# Create full reference vector aligned with DEFAULT_ELEMENTS (len = 89)
full_refs = np.zeros(len(DEFAULT_ELEMENTS), dtype=np.float32)
for el, ref in manual_refs.items():
    idx = DEFAULT_ELEMENTS.index(el)
    full_refs[idx] = ref

logger.info("Mapped manual reference energies to DEFAULT_ELEMENTS indices:")
for el in manual_refs:
    idx = DEFAULT_ELEMENTS.index(el)
    logger.info(f"  {el:>2s} (index {idx:2d}): {full_refs[idx]: .6f} eV/atom")

# Build AtomRef with full-length references
atom_ref = AtomRef(property_offset=full_refs, max_z=len(DEFAULT_ELEMENTS))


# ===============================================================
#  MODEL SETUP
# ===============================================================
model = M3GNet(element_types=element_types, is_intensive=False)


# ===============================================================
#  LIGHTNING MODULE (default Adam + CosineAnnealingLR)
# ===============================================================
lit_module = PotentialLightningModule(
    model=model,
    lr=1e-3,               # initial LR
    decay_steps=500,        # cosine decay period
    decay_alpha=0.5,        # eta_min = 0.5 * lr
    include_line_graph=True,
    stress_weight=0.0,      # disable stress term
    loss="huber_loss",      # robust Huber loss
    loss_params={"delta": 0.01},
    element_refs=atom_ref.property_offset.detach().cpu().numpy(),  # aligned with DEFAULT_ELEMENTS
)


# ===============================================================
#  TRAINER SETUP
# ===============================================================
logger_csv = CSVLogger("logs", name="M3GNet_training_scratch")

trainer = pl.Trainer(
    max_epochs=300,
    accelerator="gpu",
    devices=1,
    logger=logger_csv,
    inference_mode=False,
)


# ===============================================================
#  TRAINING LOOP
# ===============================================================
logger.info("=== Training Started ===")
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


# ===============================================================
#  SAVE MODEL
# ===============================================================
save_path = "./Ag-alloy_2025_RPBE_manual_refs"
lit_module.model.save(save_path)
logger.info(f"Model saved to {save_path}")
logger.info("=== Training Completed Successfully ===")


