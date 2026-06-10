#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error
from pymatgen.core import Structure
import ast
import json

# =========================
# Utility helpers
# =========================

def calculate_rmse(actual, predicted, scale_factor=1.0):
    """RMSE with optional scale (e.g., 1000 for meV)."""
    return np.sqrt(mean_squared_error(actual, predicted)) * scale_factor


def process_set(df):
    """Parse structures, energies, and forces."""
    structures = [Structure.from_dict(json.loads(s)) for s in df['Structure']]
    actual_e = df['Actual Energy'].values
    pred_e   = df['Pred Energy'].values
    actual_f = [ast.literal_eval(f) for f in df['Actual Forces']]
    pred_f   = [ast.literal_eval(f) for f in df['Pred Forces']]
    return structures, actual_e, pred_e, actual_f, pred_f


def calculate_formation_energies(structures, energies, element_energy_dict):
    """Formation energy per atom using manual reference energies (eV/atom)."""
    fe = []
    for struct, E in zip(structures, energies):
        comp = struct.composition
        miss = [el.symbol for el in comp.keys() if el.symbol not in element_energy_dict]
        if miss:
            raise ValueError(f"Missing reference energies for elements: {miss}")
        ref_sum = sum(comp[el] * element_energy_dict[el.symbol] for el in comp.keys())
        natoms = sum(comp.values())
        fe.append((E - ref_sum) / natoms)
    return np.array(fe)

# =========================
# Plot helpers
# =========================

def parity_plot_test(true_vals, pred_vals, xlabel, ylabel, save_path, font_prop, units):
    """Parity plot for TEST set only."""
    tv = np.array(true_vals)
    pv = np.array(pred_vals)

    rmse = calculate_rmse(tv, pv, scale_factor=1000)

    plt.figure(figsize=(6,6))
    plt.scatter(
        tv, pv,
        label=f'Test RMSE: {rmse:.2f} {units}',
        marker='o',
        color='salmon',
        edgecolor='red',
        alpha=0.7,
        s=180
    )

    mn = min(tv.min(), pv.min())
    mx = max(tv.max(), pv.max())
    plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)

    plt.xlabel(xlabel, fontproperties=font_prop, fontsize=20)
    plt.ylabel(ylabel, fontproperties=font_prop, fontsize=20)

    leg = plt.legend(prop=font_prop)
    plt.setp(leg.get_texts(), fontsize=16)

    plt.xticks(fontproperties=font_prop, fontsize=18)
    plt.yticks(fontproperties=font_prop, fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()

# =========================
# Main
# =========================

def main():

    # ---- Font ----
    font_path = '/home/biswasm/Arial_Narrow/arialnarrow.ttf'
    font_prop = FontProperties(fname=font_path)

    # ---- Manual elemental reference energies (eV/atom) ----
    manual_refs = {
        "Ag": -2.185981,
        "Au": -2.623971,
        "Bi": -3.446913,
        "Cu": -3.248735,
        "O":  -1.758624,
        "H":  -3.492632,
        "Pd": -4.629983,
        "Sn": -3.320969
    }

    # ---- Load TEST CSV only ----
    test_df = pd.read_csv('./test_eval.csv')
    print(f"Loaded TEST set with {len(test_df)} structures")

    # ---- Parse test data ----
    structures, actual_E, pred_E, actual_F, pred_F = process_set(test_df)

    # ---- Formation energies ----
    actual_fe = calculate_formation_energies(structures, actual_E, manual_refs)
    pred_fe   = calculate_formation_energies(structures, pred_E, manual_refs)

    # ---- Energy parity plot (TEST ONLY) ----
    parity_plot_test(
        actual_fe,
        pred_fe,
        xlabel='DFT ΔE$_f$ (eV/atom)',
        ylabel='Predicted ΔE$_f$ (eV/atom)',
        save_path='Energy_Test_Only.png',
        font_prop=font_prop,
        units='meV/atom'
    )

    # ---- Flatten forces ----
    actual_forces = []
    pred_forces   = []

    for af_struct, pf_struct in zip(actual_F, pred_F):
        for af_atom, pf_atom in zip(af_struct, pf_struct):
            for af_comp, pf_comp in zip(af_atom, pf_atom):
                actual_forces.append(af_comp)
                pred_forces.append(pf_comp)

    actual_forces = np.array(actual_forces)
    pred_forces   = np.array(pred_forces)

    rmse_f = calculate_rmse(actual_forces, pred_forces, scale_factor=1000)

    # ---- Force parity plot (TEST ONLY) ----
    plt.figure(figsize=(6,6))
    plt.scatter(
        actual_forces, pred_forces,
        label=f'Test RMSE: {rmse_f:.2f} meV/Å',
        marker='o',
        color='salmon',
        edgecolor='red',
        alpha=0.7,
        s=180
    )

    mn = min(actual_forces.min(), pred_forces.min())
    mx = max(actual_forces.max(), pred_forces.max())
    plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)

    plt.xlabel('DFT Forces (eV/Å)', fontproperties=font_prop, fontsize=20)
    plt.ylabel('Predicted Forces (eV/Å)', fontproperties=font_prop, fontsize=20)

    leg = plt.legend(prop=font_prop)
    plt.setp(leg.get_texts(), fontsize=16)

    plt.xticks(fontproperties=font_prop, fontsize=18)
    plt.yticks(fontproperties=font_prop, fontsize=18)

    plt.tight_layout()
    plt.savefig('Forces_Test_Only.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()

    # ---- Print statistics ----
    print("\nTEST SET STATISTICS")
    print(f"Formation Energy RMSE: {calculate_rmse(actual_fe, pred_fe, 1000):.2f} meV")
    print(f"Forces RMSE:           {rmse_f:.2f} meV/Å")


if __name__ == "__main__":
    main()
