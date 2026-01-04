#!/usr/bin/env python
"""
MSEP Training Script
====================
Train the MSEP model on QM9 data and save to a pickle file.

This script should be run ONCE by the developer to generate the model file.
End users do NOT need to run this script - they only need msep_model.pkl.

Usage:
    python msep_train.py
    
Output:
    msep_model.pkl - The trained model file to distribute with MSEP
"""

import io
import sys
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import requests

from rdkit import Chem
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations
import math
from rdkit.Chem import (
    AllChem, Descriptors, rdMolDescriptors, Crippen,
    rdchem, GetSymmSSSR, rdFingerprintGenerator
)
from rdkit.Chem.rdchem import HybridizationType, BondType

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, PolynomialFeatures

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_SIZE = 50000
RANDOM_STATE = 42
OUTPUT_MODEL_PATH = 'msep_model.pkl'

# =============================================================================
# IMPORT CORE MODULE (for feature extraction functions)
# =============================================================================

# Import all the feature extraction functions from msep_core
from msep_core import (
    # Constants
    HARTREE_TO_KCAL, KCAL_TO_HARTREE, ATOMIC_ENERGIES_B3LYP,
    FP_SIZE, FP_GEN,
    # Functions
    is_valid_smiles, count_atoms, compute_atomic_baseline,
    extract_all_features, get_fingerprint,
)

# =============================================================================
# DATA LOADING
# =============================================================================

def get_heavy_atom_count(smiles: str) -> int:
    """Get number of heavy atoms from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def load_qm9_dataset() -> pd.DataFrame:
    """Load QM9 dataset from URL or local file."""
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    print("\nLoading QM9 dataset...")
    sys.stdout.flush()
    
    try:
        print(f"    Downloading from {url}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        print(f"    Download successful: {len(df)} rows")
    except Exception as e:
        print(f"    Download failed: {e}")
        print("    Trying local qm9.csv...")
        try:
            df = pd.read_csv("qm9.csv")
            print(f"    Local file loaded: {len(df)} rows")
        except Exception as e2:
            raise RuntimeError(f"Could not load QM9 data: {e2}")
    
    # Filter valid SMILES
    print("    Validating SMILES...")
    sys.stdout.flush()
    
    valid_mask = df['smiles'].apply(is_valid_smiles)
    df = df[valid_mask].copy()
    print(f"    Valid molecules: {len(df)}")
    
    # Add useful columns
    print("    Computing derived columns...")
    sys.stdout.flush()
    
    df['n_heavy'] = df['smiles'].apply(get_heavy_atom_count)
    
    return df


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model():
    """Train the MSEP model and save to pickle file."""
    
    print("=" * 75)
    print("MSEP MODEL TRAINING")
    print("=" * 75)
    print(f"\nConfiguration:")
    print(f"    Training size: {TRAINING_SIZE:,}")
    print(f"    Output: {OUTPUT_MODEL_PATH}")
    print(f"    All calculations in HARTREE")
    sys.stdout.flush()
    
    # Load data
    qm9_df = load_qm9_dataset()
    
    # Show size distribution
    size_counts = qm9_df['n_heavy'].value_counts().sort_index()
    total = len(qm9_df)
    cumulative = 0
    print("\n    Size distribution:")
    for n_heavy in sorted(size_counts.index):
        count = size_counts[n_heavy]
        pct = 100 * count / total
        cumulative += pct
        print(f"        N={n_heavy:2d}: {count:6,} ({pct:5.1f}%, cumulative: {cumulative:5.1f}%)")
    
    print("\n" + "=" * 75)
    print("QM9 Data LOADED")
    print("=" * 75)
    sys.stdout.flush()
    
    has_zpve = 'zpve' in qm9_df.columns
    print(f"\n    QM9 has ZPVE column: {has_zpve}")
    
    # =========================================================================
    # PHASE 1: DATA PREPARATION
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 1: Data Preparation")
    print("=" * 75)
    
    np.random.seed(RANDOM_STATE)
    sample_df = qm9_df.sample(n=TRAINING_SIZE, random_state=RANDOM_STATE).copy()
    
    print(f"\nExtracting features from {TRAINING_SIZE:,} molecules...")
    start_time = time.time()
    
    feature_dicts = []
    metadata_list = []
    u0_list = []
    zpve_list = []
    baseline_list = []
    n_heavy_list = []
    n_atoms_list = []
    n_H_list = []
    n_bonds_list = []
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        if i % 5000 == 0:
            print(f"    {i:,}/{TRAINING_SIZE:,}")
        
        features, metadata = extract_all_features(row['smiles'])
        if features is None:
            continue
        
        feature_dicts.append(features)
        metadata_list.append(metadata)
        u0_list.append(row['u0'])
        baseline_list.append(metadata['atomic_baseline'])
        n_heavy_list.append(metadata['n_heavy'])
        n_atoms_list.append(metadata['n_atoms'])
        n_H_list.append(metadata['atom_counts'].get('H', 0))
        n_bonds_list.append(features.get('n_bonds', 0))
        
        if has_zpve:
            zpve_list.append(row['zpve'])
        else:
            zpve_list.append(features.get('zpve_estimated', 0.05))
    
    elapsed = time.time() - start_time
    print(f"    Done in {elapsed:.1f}s")
    
    feature_names = sorted(feature_dicts[0].keys())
    n_features = len(feature_names)
    
    X = np.array([[fd.get(fn, 0.0) for fn in feature_names] for fd in feature_dicts])
    y_u0 = np.array(u0_list)
    y_zpve = np.array(zpve_list)
    baselines = np.array(baseline_list)
    n_heavy = np.array(n_heavy_list)
    n_atoms = np.array(n_atoms_list)
    n_H = np.array(n_H_list)
    n_bonds = np.array(n_bonds_list)
    
    y_formation = y_u0 - baselines
    
    print(f"\n    Samples: {len(X):,}")
    print(f"    Features: {n_features}")
    sys.stdout.flush()
    
    # =========================================================================
    # PHASE 2: ZPVE MODEL
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 2: ZPVE Model")
    print("=" * 75)
    
    print("\n    Fitting multi-component ZPVE model...")
    
    X_zpve = np.column_stack([
        n_atoms,
        n_H,
        n_atoms ** 2,
        n_bonds,
        np.ones(len(n_atoms)),
    ])
    
    from sklearn.linear_model import Ridge as RidgeReg
    zpve_linear_model = RidgeReg(alpha=0.1)
    zpve_linear_model.fit(X_zpve, y_zpve)
    
    coef = zpve_linear_model.coef_
    intercept_term = coef[4]
    
    print(f"\n    ZPVE = {coef[0]:.6f} × N_atoms")
    print(f"         + {coef[1]:.6f} × N_H")
    print(f"         + {coef[2]:.8f} × N_atoms²")
    print(f"         + {coef[3]:.6f} × N_bonds")
    print(f"         + {intercept_term:.6f}")
    
    ZPVE_COEFFS = {
        'n_atoms': coef[0],
        'n_H': coef[1],
        'n_atoms_sq': coef[2],
        'n_bonds': coef[3],
        'intercept': intercept_term,
    }
    
    zpve_pred_linear = zpve_linear_model.predict(X_zpve)
    zpve_linear_mae = np.mean(np.abs(y_zpve - zpve_pred_linear))
    print(f"\n    Linear model MAE: {zpve_linear_mae:.6f} Ha ({zpve_linear_mae * HARTREE_TO_KCAL:.2f} kcal/mol)")
    sys.stdout.flush()
    
    # =========================================================================
    # PHASE 3: ZPVE RESIDUAL MODEL
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 3: ZPVE Residual Model")
    print("=" * 75)
    
    scaler_main = RobustScaler()
    X_scaled = scaler_main.fit_transform(X)
    X_scaled = np.clip(np.nan_to_num(X_scaled), -10, 10)
    
    zpve_residual = y_zpve - zpve_pred_linear
    
    print(f"\n    Residual range: [{zpve_residual.min():.6f}, {zpve_residual.max():.6f}] Ha")
    print(f"    Residual std: {zpve_residual.std():.6f} Ha")
    
    print("\n[ZPVE Residual GB Model]")
    zpve_residual_model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.015,
        l2_regularization=0.2,
        min_samples_leaf=10,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )
    zpve_residual_model.fit(X_scaled, zpve_residual)
    
    zpve_residual_pred = zpve_residual_model.predict(X_scaled)
    zpve_total_pred = zpve_pred_linear + zpve_residual_pred
    
    zpve_final_error = y_zpve - zpve_total_pred
    zpve_mae = np.mean(np.abs(zpve_final_error))
    zpve_max = np.max(np.abs(zpve_final_error))
    
    print(f"    Iterations: {zpve_residual_model.n_iter_}")
    print(f"    Final ZPVE MAE: {zpve_mae:.6f} Ha ({zpve_mae * HARTREE_TO_KCAL:.4f} kcal/mol)")
    print(f"    Final ZPVE Max: {zpve_max:.6f} Ha ({zpve_max * HARTREE_TO_KCAL:.4f} kcal/mol)")
    
    ZPVE_PARAMS = {
        'coeffs': ZPVE_COEFFS,
        'mae': zpve_mae,
        'max': zpve_max,
    }
    sys.stdout.flush()
    
    # =========================================================================
    # PHASE 4: FORMATION ENERGY MODEL
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 4: Formation Energy Model")
    print("=" * 75)
    
    print("\n[Extracting fingerprints]")
    fp_list = []
    for meta in metadata_list:
        fp = get_fingerprint(meta['smiles'])
        fp_list.append(fp if fp is not None else np.zeros(FP_SIZE))
    X_fp = np.array(fp_list)
    
    X_full = np.hstack([X_scaled, X_fp])
    print(f"    Full feature dimension: {X_full.shape[1]}")
    
    scaler_ml = RobustScaler()
    X_ml = scaler_ml.fit_transform(X_full)
    X_ml = np.clip(np.nan_to_num(X_ml), -10, 10)
    
    # Level 1: Huber
    print("\n[Level 1: Huber Regression]")
    huber = HuberRegressor(epsilon=1.35, max_iter=500, alpha=0.001)
    huber.fit(X_ml, y_formation)
    pred_huber = huber.predict(X_ml)
    res_huber = y_formation - pred_huber
    mae_huber = np.mean(np.abs(res_huber))
    print(f"    MAE: {mae_huber:.6f} Ha ({mae_huber * HARTREE_TO_KCAL:.4f} kcal/mol)")
    
    # Level 2: Ridge + Poly
    print("\n[Level 2: Ridge + Polynomial]")
    importance = np.abs(huber.coef_)
    n_top = min(40, len(importance))
    top_idx = np.argsort(importance)[-n_top:]
    X_top = X_ml[:, top_idx]
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X_top)
    X_enhanced = np.hstack([X_ml, X_poly])
    print(f"    Enhanced features: {X_enhanced.shape[1]}")
    
    ridge = Ridge(alpha=0.5, random_state=RANDOM_STATE)
    ridge.fit(X_enhanced, y_formation)
    pred_ridge = ridge.predict(X_enhanced)
    res_ridge = y_formation - pred_ridge
    mae_ridge = np.mean(np.abs(res_ridge))
    print(f"    MAE: {mae_ridge:.6f} Ha ({mae_ridge * HARTREE_TO_KCAL:.4f} kcal/mol)")
    
    # Level 3: Main GB
    print("\n[Level 3: Main Gradient Boosting]")
    gb_main = HistGradientBoostingRegressor(
        max_iter=1500,
        max_depth=12,
        learning_rate=0.012,
        l2_regularization=0.08,
        min_samples_leaf=4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=RANDOM_STATE,
    )
    gb_main.fit(X_enhanced, res_ridge)
    pred_gb = gb_main.predict(X_enhanced)
    res_gb = res_ridge - pred_gb
    mae_gb = np.mean(np.abs(res_gb))
    print(f"    Iterations: {gb_main.n_iter_}")
    print(f"    MAE: {mae_gb:.6f} Ha ({mae_gb * HARTREE_TO_KCAL:.4f} kcal/mol)")
    
    # Level 4: Refinement
    print("\n[Level 4: Refinement GB]")
    gb_refine = HistGradientBoostingRegressor(
        max_iter=800,
        max_depth=8,
        learning_rate=0.008,
        l2_regularization=0.2,
        min_samples_leaf=6,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE + 1,
    )
    gb_refine.fit(X_enhanced, res_gb)
    pred_refine = gb_refine.predict(X_enhanced)
    print(f"    Iterations: {gb_refine.n_iter_}")
    
    pred_formation_ml = pred_ridge + pred_gb + pred_refine
    sys.stdout.flush()
    
    # =========================================================================
    # PHASE 5: SIZE AND FG CORRECTIONS
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 5: Size and FG Corrections")
    print("=" * 75)
    
    res_for_corr = y_formation - pred_formation_ml
    
    print("\n[Size Correction]")
    size_residuals = defaultdict(list)
    for i, n in enumerate(n_heavy):
        size_residuals[int(n)].append(res_for_corr[i])
    
    sizes = sorted([s for s in size_residuals.keys() if len(size_residuals[s]) >= 50])
    means = [np.mean(size_residuals[s]) for s in sizes]
    
    SIZE_COEFFS = np.polyfit(sizes, means, 2) if len(sizes) >= 3 else [0, 0, 0]
    size_corr = np.polyval(SIZE_COEFFS, n_heavy)
    
    print("\n[FG Corrections]")
    res_after_size = res_for_corr - size_corr
    
    fg_residuals = defaultdict(list)
    for i, meta in enumerate(metadata_list):
        for fg, count in meta['fg_counts'].items():
            if count > 0:
                fg_residuals[fg].append(res_after_size[i])
    
    FG_CORRECTIONS = {}
    for fg, residuals in sorted(fg_residuals.items(), key=lambda x: -len(x[1])):
        if len(residuals) >= 100:
            bias = np.mean(residuals)
            if abs(bias * HARTREE_TO_KCAL) > 0.15:
                FG_CORRECTIONS[fg] = bias
    
    print(f"    Total FG corrections: {len(FG_CORRECTIONS)}")
    
    fg_corr = np.zeros(len(X))
    for i, meta in enumerate(metadata_list):
        for fg, count in meta['fg_counts'].items():
            if fg in FG_CORRECTIONS and count > 0:
                fg_corr[i] += FG_CORRECTIONS[fg]
    
    sys.stdout.flush()
    
    # =========================================================================
    # PHASE 6: FINAL EVALUATION
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 6: Final Evaluation")
    print("=" * 75)
    
    pred_formation_total = pred_formation_ml + size_corr + fg_corr
    pred_u0 = baselines + pred_formation_total
    
    u0_errors = y_u0 - pred_u0
    u0_mae = np.mean(np.abs(u0_errors))
    u0_max = np.max(np.abs(u0_errors))
    
    print("\n    U0 MODEL:")
    print(f"        MAE: {u0_mae:.6f} Ha ({u0_mae * HARTREE_TO_KCAL:.4f} kcal/mol)")
    print(f"        Max: {u0_max:.6f} Ha ({u0_max * HARTREE_TO_KCAL:.4f} kcal/mol)")
    
    print("\n    ZPVE MODEL:")
    print(f"        MAE: {ZPVE_PARAMS['mae']:.6f} Ha ({ZPVE_PARAMS['mae'] * HARTREE_TO_KCAL:.4f} kcal/mol)")
    
    pred_e_elec = pred_u0 - zpve_total_pred
    actual_e_elec = y_u0 - y_zpve
    e_elec_errors = pred_e_elec - actual_e_elec
    e_elec_mae = np.mean(np.abs(e_elec_errors))
    
    print("\n    E_elec (U0 - ZPVE):")
    print(f"        MAE: {e_elec_mae:.6f} Ha ({e_elec_mae * HARTREE_TO_KCAL:.4f} kcal/mol)")
    
    sys.stdout.flush()
    
    # =========================================================================
    # PHASE 7: SAVE MODEL
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 7: Saving Model")
    print("=" * 75)
    
    MODEL_COMPONENTS = {
        'version': 'msep_v1.0',
        'scaler_main': scaler_main,
        'scaler_ml': scaler_ml,
        'poly': poly,
        'top_idx': top_idx,
        'huber': huber,
        'ridge': ridge,
        'gb_main': gb_main,
        'gb_refine': gb_refine,
        # ZPVE model components
        'zpve_coeffs': ZPVE_COEFFS,
        'zpve_linear_model': zpve_linear_model,
        'zpve_residual_model': zpve_residual_model,
        # Corrections
        'size_coeffs': SIZE_COEFFS,
        'fg_corrections': FG_CORRECTIONS,
    }
    
    MODEL_PARAMS = {
        'feature_names': feature_names,
        'n_features': n_features,
        'u0_mae': u0_mae,
        'u0_mae_kcal': u0_mae * HARTREE_TO_KCAL,
        'zpve_mae': ZPVE_PARAMS['mae'],
        'zpve_mae_kcal': ZPVE_PARAMS['mae'] * HARTREE_TO_KCAL,
        'zpve_coeffs': ZPVE_COEFFS,
        'e_elec_mae': e_elec_mae,
        'training_size': TRAINING_SIZE,
    }
    
    # Save to pickle
    saved_data = {
        'MODEL_COMPONENTS': MODEL_COMPONENTS,
        'MODEL_PARAMS': MODEL_PARAMS,
        'feature_names': feature_names,
    }
    
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(saved_data, f)
    
    file_size = os.path.getsize(OUTPUT_MODEL_PATH) / (1024 * 1024)
    
    print(f"\n    Model saved to: {OUTPUT_MODEL_PATH}")
    print(f"    File size: {file_size:.1f} MB")
    print(f"    ZPVE formula: {ZPVE_COEFFS['n_atoms']:.5f}×N + {ZPVE_COEFFS['n_H']:.5f}×H + ...")
    
    # =========================================================================
    # TEST THE SAVED MODEL
    # =========================================================================
    
    print("\n" + "=" * 75)
    print("PHASE 8: Testing Saved Model")
    print("=" * 75)
    
    # Import and test
    import msep_core
    msep_core.load_model(OUTPUT_MODEL_PATH)
    
    test_molecules = [
        ("C", "methane"),
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("CC(=O)O", "acetic_acid"),
    ]
    
    print(f"\n{'Molecule':<15} {'SMILES':<15} {'SCF (Ha)':<18} {'Uncertainty':<12}")
    print("-" * 60)
    
    for smi, name in test_molecules:
        result = msep_core.predict_molecule(smi, verbose=False)
        if result:
            print(f"{name:<15} {smi:<15} {result['scf_solvated']:<18.8f} ±{result['uncertainty']:.6f}")
    
    print("\n" + "=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)
    print(f"\nDistribute these files to users:")
    print(f"    1. msep_core.py      - Core library")
    print(f"    2. {OUTPUT_MODEL_PATH}  - Trained model")
    print(f"    3. msep_predict.py   - Prediction script")
    
    return MODEL_COMPONENTS, MODEL_PARAMS


if __name__ == '__main__':
    train_model()
