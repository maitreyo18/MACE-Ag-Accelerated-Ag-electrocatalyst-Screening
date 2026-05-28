import pandas as pd
import ast
from pymatgen.core import Structure
import torch
import warnings
import matgl
import dgl
import os
from tqdm import tqdm

df_filtered = pd.read_csv('PBE-HSE+SOC-HaPs.csv')



# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OUTPUT_PT = "Vectors_HaP.pt"

os.environ['MATGL_BACKEND'] = 'DGL'
matgl.set_backend("DGL")

# --------------------------------------------------
# LOAD MODEL (CPU, SAFE)
# --------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_wrapper = matgl.load_model("M3GNet-MatPES-PBE-v2025.1-PES")
    model = model_wrapper.model
    model.eval()

# --------------------------------------------------
# STORAGE DICT
# --------------------------------------------------
vectors_dict = {}

# --------------------------------------------------
# MAIN LOOP (WITH tqdm)
# --------------------------------------------------
for idx, row in tqdm(
    df_filtered.iterrows(),
    total=len(df_filtered),
    desc="Extracting M3GNet GC3 embeddings"
):
    try:
        # --- Parse structure ---
        struct_dict = ast.literal_eval(row["Structure"])
        structure = Structure.from_dict(struct_dict)

        # --- Predict + extract features ---
        with torch.no_grad():
            prediction = model.predict_structure(
                structure, return_features=True
            )

        # --- GC3 features (EXACTLY like your worker code) ---
        gc3_node_vecs = prediction["gc_3"]["node_feat"]  
        gc3_edge_vecs = prediction["gc_3"]["edge_feat"]  

        # --- Mean pooling ---
        mean_node = torch.mean(gc3_node_vecs, dim=0)
        mean_edge = torch.mean(gc3_edge_vecs, dim=0)

        # --- Concatenate node + edge ---
        total_vecs = torch.cat((mean_node, mean_edge))   

        # --- Store ---
        vectors_dict[int(idx)] = {
            "Composition": row["Compound Name"],
            "Vector": total_vecs.cpu().numpy().tolist(),
            "Fidelity": row["Functional"],
            "Band_gap": float(row["Band Gap"]),
        }

    except Exception as e:
        print(f"[WARN] Failed at index {idx}: {e}")

# --------------------------------------------------
# SAVE
# --------------------------------------------------
torch.save(vectors_dict, OUTPUT_PT)

print(f"✔ Saved {len(vectors_dict)} embeddings to {OUTPUT_PT}")