#!/usr/bin/env python
"""
Extract per-atom invariant (scalar) embeddings from a MACE model
for all .vasp files in ./structs and save as mace_embeds.pt
"""

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pathlib import Path

import torch
import ase.io

from mace.data import AtomicData, utils as mace_data_utils
from mace.tools import utils as mace_utils
from mace.tools import torch_geometric

# ── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = "/depot/amannodi/data/Maitreyo_calcs/Ag-alloy_RPBE/MLIPs/1_3_ML/MACE_ZBL/MACE_models_scratch/MACE_alloy+adsorb_stagetwo_compiled.model"  # <-- put your model path here
STRUCTS_DIR = Path("./structs")
OUTPUT_PATH = Path("./mace_embeds.pt")
DEVICE = "cuda"
# ────────────────────────────────────────────────────────────────────


def load_model(path, device):
    try:
        model = torch.jit.load(path, map_location=device)
    except Exception:
        model = torch.load(path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def extract_invariant(x, num_layers, num_features, l_max):
    out = [x[:, :num_features]]
    for i in range(1, num_layers):
        start = i * (l_max + 1) ** 2 * num_features
        out.append(x[:, start : start + num_features])
    return torch.cat(out, dim=-1)


def get_embeddings(model, atoms, z_table, r_max, heads, device,
                   num_interactions, num_scalar_features, l_max):
    keyspec = mace_data_utils.KeySpecification()
    config = mace_data_utils.config_from_atoms(atoms, key_specification=keyspec)
    atomic_data = AtomicData.from_config(
        config, z_table=z_table, cutoff=r_max, heads=heads
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(data_loader)).to(device)

    # Cast batch tensors to match model dtype (float64)
    batch_dict = batch.to_dict()
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            batch_dict[k] = v.to(torch.float64)
    # positions needs grad for autograd-based force computation inside the model
    batch_dict["positions"].requires_grad_(True)
    output = model(batch_dict)

    node_feats = output["node_feats"]
    invariants = extract_invariant(node_feats, num_interactions, num_scalar_features, l_max)
    return invariants.cpu()


def infer_irreps_from_buffers(model):
    """Infer num_scalar_features and l_max from product layer output_mask shapes,
    since TorchScript models don't allow Python-level attribute indexing."""
    output_masks = {}
    for name, buf in model.named_buffers():
        if "products." in name and "linear.output_mask" in name:
            # e.g. "products.0.linear.output_mask" -> layer 0
            layer_idx = int(name.split(".")[1])
            output_masks[layer_idx] = buf.shape[0]

    last_layer = max(output_masks)
    num_scalar_features = output_masks[last_layer]  # last layer is scalars only

    first_layer_dim = output_masks[0]
    # first_layer_dim = num_scalar_features * (l_max+1)^2
    l_max_sq = first_layer_dim // num_scalar_features
    l_max = int(round(l_max_sq ** 0.5)) - 1

    return num_scalar_features, l_max


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH, DEVICE)

    # Model metadata
    r_max = float(model.r_max)
    z_table = mace_utils.AtomicNumberTable(
        [int(z) for z in model.atomic_numbers.tolist()]
    )
    num_interactions = int(model.num_interactions)
    num_scalar_features, l_max = infer_irreps_from_buffers(model)

    try:
        heads = model.heads
    except Exception:
        heads = ["Default"]

    print(f"  num_interactions = {num_interactions}")
    print(f"  l_max = {l_max}")
    print(f"  scalar features/layer = {num_scalar_features}")
    print(f"  embedding dim = {num_interactions * num_scalar_features}")
    print()

    # Collect .vasp files
    vasp_files = sorted(STRUCTS_DIR.glob("*.vasp"))
    print(f"Found {len(vasp_files)} .vasp files in {STRUCTS_DIR}\n")

    results = {}
    for idx, fpath in enumerate(vasp_files):
        atoms = ase.io.read(str(fpath), format="vasp")
        embeds = get_embeddings(
            model, atoms, z_table, r_max, heads, DEVICE,
            num_interactions, num_scalar_features, l_max,
        )
        results[idx] = {
            "file_name": fpath.name,
            "embeds": embeds,
        }
        if (idx + 1) % 50 == 0 or idx == len(vasp_files) - 1:
            print(f"  [{idx+1}/{len(vasp_files)}] {fpath.name} -> {embeds.shape}")

    torch.save(results, OUTPUT_PATH)
    print(f"\nSaved {len(results)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
