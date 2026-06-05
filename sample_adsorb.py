import os
import logging
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
IN_CSV = "OH-adsorb_Ag_1_3_3x3x1.csv"
OUT_DIR = "./OH-adsorb_Sampled_2_step_35_3x3x1_1_3_ML"
OUT_CSV = os.path.join(OUT_DIR, "OH-adsorb_Ag_1_3_3x3x1_sampled_35.csv")
LOG_PATH = os.path.join(OUT_DIR, "sampling_OH.log")

MAX_KEEP_STEP1 = 15  # linear step
MAX_KEEP_STEP2 = 20  # quadratic step
MAX_KEEP = MAX_KEEP_STEP1 + MAX_KEEP_STEP2

# ============================================================
# LOGGING SETUP
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)
logger = logging.getLogger("sampler")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fmt)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info("=== Start sampling (2-step: linear+quadratic) ===")
logger.info(f"Input CSV: {IN_CSV}")
logger.info(f"Output CSV: {OUT_CSV}")
logger.info(f"Sampling scheme: {MAX_KEEP_STEP1} linear + {MAX_KEEP_STEP2} quadratic = {MAX_KEEP}")

# ============================================================
# LOAD CSV
# ============================================================
df = pd.read_csv(IN_CSV)

required_cols = ["Alloy_name", "Energy"]
for col in required_cols:
    if col not in df.columns:
        logger.error(f"CSV must contain column: {col}")
        raise SystemExit(1)

df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")

# ----
# Parse base name and index from Alloy_name (same _interm_xx pattern)
# ----
m = df["Alloy_name"].str.extract(r"^(?P<base>.+_interm)_(?P<idx>\d+)$")
df["base"] = m["base"]
df["interm_idx"] = pd.to_numeric(m["idx"], errors="coerce")

before_rows = len(df)
df = df.dropna(subset=["base", "interm_idx", "Energy"]).copy()
df["interm_idx"] = df["interm_idx"].astype(int)

logger.info(f"Loaded rows: {before_rows} | usable rows after parsing: {len(df)}")

# ============================================================
# HELPER FUNCTION: ENERGY-BASED SAMPLING
# ============================================================
def sample_energy_based(group, n, exclude=None, force_ground=False, mode="linear"):
    """Energy-based sampling from a group."""
    g = group.sort_values("Energy").copy()
    g["orig_index"] = g.index
    g = g.reset_index(drop=True)

    energies = g["Energy"].values
    minE, maxE = energies.min(), energies.max()
    chosen_positions = []
    used_positions = set()

    if exclude:
        exclude_positions = g[g["orig_index"].isin(exclude)].index.tolist()
        used_positions.update(exclude_positions)

    if len(g) <= n:
        return g["orig_index"].tolist()

    for i in range(n):
        frac = i / (n - 1) if n > 1 else 0
        if mode == "linear":
            weight = frac
        elif mode == "quadratic":
            weight = frac ** 2
        else:
            raise ValueError("mode must be 'linear' or 'quadratic'")
        target = minE + weight * (maxE - minE)
        diffs = np.abs(energies - target)
        for u in used_positions:
            if u < len(diffs):
                diffs[u] = np.inf
        pos = diffs.argmin()
        chosen_positions.append(pos)
        used_positions.add(pos)

    if force_ground:
        ground_pos = g["interm_idx"].idxmax()  # ensure ground state is kept
        if ground_pos not in chosen_positions:
            chosen_positions[-1] = ground_pos

    chosen = g.iloc[chosen_positions]["orig_index"].tolist()
    return chosen

# ============================================================
# MAIN SAMPLING LOOP
# ============================================================
kept_idx = []

for base, g in df.groupby("base", sort=False):
    logger.info(f"[GROUP] {base} | total items: {len(g)}")

    if len(g) <= MAX_KEEP:
        logger.info(f" Keeping all {len(g)} intermediates (<= {MAX_KEEP})")
        kept_idx.extend(g.index.tolist())
        continue

    # Step 1: linear (global)
    chosen1 = sample_energy_based(g, MAX_KEEP_STEP1, force_ground=True, mode="linear")

    # Step 2: quadratic (residual)
    chosen2 = sample_energy_based(g, MAX_KEEP_STEP2, exclude=chosen1, mode="quadratic")

    chosen = sorted(set(chosen1 + chosen2))
    if len(chosen) < MAX_KEEP:
        remaining = [ix for ix in g.index if ix not in chosen]
        chosen.extend(remaining[: MAX_KEEP - len(chosen)])

    kept_idx.extend(chosen)
    logger.info(f" Selected {len(chosen)} of {len(g)} intermediates (target {MAX_KEEP})")

    for row in df.loc[chosen].itertuples():
        logger.info(f" KEPT: {row.Alloy_name} | E={row.Energy:.6f}")

# ============================================================
# SAVE OUTPUT
# ============================================================
df_sampled = df.loc[kept_idx].copy()
df_sampled = df_sampled.drop(columns=["base", "interm_idx"])
df_sampled.to_csv(OUT_CSV, index=False)

logger.info(f"Saved sampled CSV: {OUT_CSV}")
logger.info("=== Done ===")