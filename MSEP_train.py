#!/usr/bin/env python
# coding: utf-8

# # Machine Learning SCF Energy Prediction (MSEP) Training Script  
# 
# ## Step 1: QM9 data feature engineering 
# 
# ## Step 2: Prediction Training 
# 

# In[ ]:


import io
import sys
import os
import time
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
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')



# =============================================================================
# CONSTANTS
# =============================================================================

HARTREE_TO_KCAL = 627.509474

# Atomic reference energies (B3LYP/6-31G(2df,p))
ATOMIC_ENERGIES_B3LYP = {
    'H': -0.60389849,
    'C': -38.07365363,
    'N': -54.74921767,
    'O': -75.22605827,
    'F': -99.87025500,
}

print("\nConstants defined:")
print(f"    HARTREE_TO_KCAL = {HARTREE_TO_KCAL}")
print("    ATOMIC_ENERGIES_B3LYP loaded for H, C, N, O, F")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string is valid."""
    if pd.isna(smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False


def get_heavy_atom_count(smiles: str) -> int:
    """Get number of heavy atoms from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def get_atomic_baseline(smiles: str) -> float:
    """Calculate atomic baseline energy from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float('nan')
    mol_h = Chem.AddHs(mol)
    energy = 0.0
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ATOMIC_ENERGIES_B3LYP:
            energy += ATOMIC_ENERGIES_B3LYP[sym]
        else:
            return float('nan')  # Unsupported element
    return energy


# =============================================================================
# DATA LOADING
# =============================================================================

def load_qm9_dataset() -> pd.DataFrame:
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
# MAIN EXECUTION
# =============================================================================

qm9_df = load_qm9_dataset()

print("\n" + "-" * 75)
print("QM9 Dataset Statistics")
print("-" * 75)

print(f"\nTotal molecules: {len(qm9_df):,}")
print(f"\nU0 (internal energy at 0K):")
print(f"    Min:  {qm9_df['u0'].min():.4f} Ha ({qm9_df['u0'].min() * HARTREE_TO_KCAL:.2f} kcal/mol)")
print(f"    Max:  {qm9_df['u0'].max():.4f} Ha ({qm9_df['u0'].max() * HARTREE_TO_KCAL:.2f} kcal/mol)")
print(f"    Mean: {qm9_df['u0'].mean():.4f} Ha ({qm9_df['u0'].mean() * HARTREE_TO_KCAL:.2f} kcal/mol)")
print(f"    Std:  {qm9_df['u0'].std():.4f} Ha ({qm9_df['u0'].std() * HARTREE_TO_KCAL:.2f} kcal/mol)")

print(f"\nHeavy atom distribution:")
size_counts = qm9_df['n_heavy'].value_counts().sort_index()
total = len(qm9_df)
cumulative = 0
for n_heavy in sorted(size_counts.index):
    count = size_counts[n_heavy]
    pct = 100 * count / total
    cumulative += pct
    print(f"    N={n_heavy:2d}: {count:6,} ({pct:5.1f}%, cumulative: {cumulative:5.1f}%)")

print(f"\nAvailable columns: {list(qm9_df.columns)}")

print("\nSample molecules:")
sample = qm9_df.sample(n=5, random_state=42)[['smiles', 'n_heavy', 'u0']]
for _, row in sample.iterrows():
    print(f"    {row['smiles']:20s} | N={row['n_heavy']:2d} | U0={row['u0']:.4f} Ha")

print("\n" + "=" * 75)
print("QM9 Data LOADED")
print("=" * 75)
sys.stdout.flush()


# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================
HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.2114
BOHR_TO_ANGSTROM = 0.529177
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
# Boltzmann constant in Ha/K
KB_HARTREE = 3.1668e-6

# =============================================================================
# ATOMIC ENERGIES (B3LYP/6-31G(2df,p) from QM9)
# =============================================================================
ATOMIC_ENERGIES_B3LYP = {
    'H': -0.500273,    
    'C': -37.846772,
    'N': -54.583861,
    'O': -75.064579,
    'F': -99.718730,
}

# =============================================================================
# HÜCKEL THEORY PARAMETERS
# =============================================================================

# Coulomb integrals α
HUCKEL_ALPHA = {
    'C': 0.0,      
    'N': 0.5,      
    'O': 1.0,      
    'F': 1.5,     
    'S': 0.0,      
}
# Resonance integrals β (in units of β_CC = 1)
# These represent the interaction between adjacent p-orbitals
HUCKEL_BETA = {
    ('C', 'C'): 1.0,    
    ('C', 'N'): 1.0,     
    ('C', 'O'): 0.8,     
    ('N', 'N'): 0.9,     
    ('N', 'O'): 0.8,
    ('O', 'O'): 0.7,
    ('C', 'F'): 0.7,     
}
# Heteroatom corrections for different hybridizations
# h_X = (α_X - α_C) / β_CC
HETEROATOM_H = {
    'N_sp2': 0.5,    
    'N_sp3': 1.5,    
    'N_pyrrole': 1.5, 
    'O_sp2': 1.0,    
    'O_sp3': 2.0,    
    'F': 3.0,        
}
# k_XY = β_XY / β_CC (bond integral ratios)
BOND_K = {
    ('C', 'N'): 1.0,
    ('C', 'O'): 0.8,
    ('C', 'F'): 0.7,
    ('N', 'N'): 0.9,
    ('N', 'O'): 0.8,
}
print(f"\nHückel parameters loaded:")
print(f"    α values: {len(HUCKEL_ALPHA)} elements")
print(f"    β values: {len(HUCKEL_BETA)} bonds")

# =============================================================================
# EXTENDED HÜCKEL PARAMETERS
# =============================================================================

VOIP = {
    'H_1s': -13.6,
    'C_2s': -21.4,
    'C_2p': -11.4,
    'N_2s': -26.0,
    'N_2p': -13.4,
    'O_2s': -32.3,
    'O_2p': -14.8,
    'F_2s': -40.0,
    'F_2p': -18.1,
}

SLATER_EXPONENTS = {
    'H_1s': 1.30,
    'C_2s': 1.625,
    'C_2p': 1.625,
    'N_2s': 1.95,
    'N_2p': 1.95,
    'O_2s': 2.275,
    'O_2p': 2.275,
    'F_2s': 2.60,
    'F_2p': 2.60,
}

# =============================================================================
# VIBRATIONAL FREQUENCY PARAMETERS (for ZPVE)
# =============================================================================


# Used to estimate ZPVE = (1/2) * h * Σν
BOND_FREQUENCIES = {
    'C-H_stretch': 3000,
    'C-C_stretch': 1000,
    'C=C_stretch': 1650,
    'C≡C_stretch': 2200,
    'C-N_stretch': 1100,
    'C=N_stretch': 1650,
    'C≡N_stretch': 2250,
    'C-O_stretch': 1100,
    'C=O_stretch': 1700,
    'N-H_stretch': 3400,
    'O-H_stretch': 3600,
    'C-F_stretch': 1100,
    'N-N_stretch': 1000,
    'N=N_stretch': 1500,
    'N-O_stretch': 900,    
    'H-C-H_bend': 1400,
    'C-C-C_bend': 400,
    'H-N-H_bend': 1600,
    'C-O-H_bend': 1300,
}

CM_TO_HARTREE = 4.5563e-6
print(f"    Frequency data: {len(BOND_FREQUENCIES)} modes")

# =============================================================================
# SOLVATION PARAMETERS
# =============================================================================

DIELECTRIC = {
    'vacuum': 1.0,
    'gas': 1.0,
    'hexane': 1.88,
    'benzene': 2.27,
    'chloroform': 4.81,
    'thf': 7.58,
    'dichloromethane': 8.93,
    'acetone': 20.7,
    'ethanol': 24.3,
    'methanol': 32.7,
    'dmso': 46.7,
    'water': 78.4,
}

SURFACE_TENSION = {
    'vacuum': 0.0,
    'gas': 0.0,
    'water': 0.00007,      # ~0.044 kcal/mol/Å²
    'methanol': 0.00005,
    'ethanol': 0.00004,
    'dmso': 0.00006,
    'chloroform': 0.00004,
    'dichloromethane': 0.00004,
    'hexane': 0.00003,
}

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
}
print(f"    Solvents: {len(DIELECTRIC)} defined")

# =============================================================================
# ATOMIC PROPERTIES
# =============================================================================

ATOMIC_PROPS = {
    'H': {'Z': 1, 'covalent': 0.31, 'polarizability': 0.387, 
          'electronegativity': 2.20, 'hardness': 6.42, 'n_valence': 1,
          'n_2p': 0, 'lone_pairs': 0},
    'C': {'Z': 6, 'covalent': 0.76, 'polarizability': 1.76,
          'electronegativity': 2.55, 'hardness': 5.00, 'n_valence': 4,
          'n_2p': 2, 'lone_pairs': 0},
    'N': {'Z': 7, 'covalent': 0.71, 'polarizability': 1.10,
          'electronegativity': 3.04, 'hardness': 7.30, 'n_valence': 5,
          'n_2p': 3, 'lone_pairs': 1},
    'O': {'Z': 8, 'covalent': 0.66, 'polarizability': 0.802,
          'electronegativity': 3.44, 'hardness': 6.08, 'n_valence': 6,
          'n_2p': 4, 'lone_pairs': 2},
    'F': {'Z': 9, 'covalent': 0.57, 'polarizability': 0.557,
          'electronegativity': 3.98, 'hardness': 7.01, 'n_valence': 7,
          'n_2p': 5, 'lone_pairs': 3},
}
# =============================================================================
# FINGERPRINT GENERATOR
# =============================================================================

FP_SIZE = 256
FP_RADIUS = 2

try:
    FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
except:
    FP_GEN = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_hybridization(atom) -> str:
    hyb = atom.GetHybridization()
    if hyb == HybridizationType.SP3:
        return 'sp3'
    elif hyb == HybridizationType.SP2:
        return 'sp2'
    elif hyb == HybridizationType.SP:
        return 'sp'
    return 'sp3'

def get_bond_order(bond) -> float:
    bt = bond.GetBondType()
    if bt == BondType.SINGLE: return 1.0
    elif bt == BondType.DOUBLE: return 2.0
    elif bt == BondType.TRIPLE: return 3.0
    elif bt == BondType.AROMATIC: return 1.5
    return 1.0
    
def count_atoms(smiles: str) -> Optional[Dict[str, int]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
        else:
            return None
    return counts

def compute_atomic_baseline(atom_counts: Dict[str, int]) -> float:
    """Compute atomic baseline energy in Ha."""
    return sum(atom_counts[elem] * ATOMIC_ENERGIES_B3LYP[elem] for elem in atom_counts)

# =============================================================================
# 1. HÜCKEL THEORY FEATURES
# =============================================================================

def compute_huckel_features(mol) -> Dict[str, float]:
    """
    Compute Hückel theory-based features for π-systems.
    
    The Hückel secular equation: |H - ES| = 0
    where H_ij = α_i if i=j (Coulomb integral)
                β_ij if i,j bonded (resonance integral)
                0 otherwise
    
    For conjugated systems, this gives delocalization energy.
    """
    features = {}
    
    # Find π-system atoms 
    pi_atoms = []
    pi_atom_idx = []
    
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        hyb = get_hybridization(atom)
        is_arom = atom.GetIsAromatic()
        
        # Include in π-system if:
        if is_arom or (hyb == 'sp2' and sym in ['C', 'N', 'O']):
            pi_atoms.append({
                'idx': atom.GetIdx(),
                'symbol': sym,
                'aromatic': is_arom,
                'hybridization': hyb,
            })
            pi_atom_idx.append(atom.GetIdx())
    
    n_pi = len(pi_atoms)
    features['huckel_n_pi_atoms'] = float(n_pi)
    
    if n_pi == 0:
        features['huckel_alpha_sum'] = 0.0
        features['huckel_beta_sum'] = 0.0
        features['huckel_delocalization'] = 0.0
        features['huckel_homo_lumo_gap'] = 0.0
        features['huckel_total_pi_energy'] = 0.0
        return features
    
    # Build Hückel matrix
    # H_ii = α + h_i * β (where h_i is heteroatom correction)
    # H_ij = k_ij * β (where k_ij is bond correction)
    
    H = np.zeros((n_pi, n_pi))
    
    # Diagonal elements (α terms)
    alpha_sum = 0.0
    for i, pa in enumerate(pi_atoms):
        sym = pa['symbol']
        h_i = HUCKEL_ALPHA.get(sym, 0.0)
        H[i, i] = h_i
        alpha_sum += h_i
    
    features['huckel_alpha_sum'] = alpha_sum
    
    # Off-diagonal elements (β terms)
    beta_sum = 0.0
    idx_map = {idx: i for i, idx in enumerate(pi_atom_idx)}
    
    for bond in mol.GetBonds():
        a1_idx = bond.GetBeginAtomIdx()
        a2_idx = bond.GetEndAtomIdx()
        
        if a1_idx in idx_map and a2_idx in idx_map:
            i = idx_map[a1_idx]
            j = idx_map[a2_idx]
            
            s1 = mol.GetAtomWithIdx(a1_idx).GetSymbol()
            s2 = mol.GetAtomWithIdx(a2_idx).GetSymbol()
            
            # Get k_ij value
            key = tuple(sorted([s1, s2]))
            k_ij = HUCKEL_BETA.get(key, 1.0)
            
            H[i, j] = k_ij
            H[j, i] = k_ij
            beta_sum += k_ij
    
    features['huckel_beta_sum'] = beta_sum
    
    # Solve eigenvalue problem
    try:
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.sort(eigenvalues)
        
        n_pi_electrons = n_pi
        
        n_filled = n_pi_electrons // 2
        
        # Total π energy (in units of β)
        # E_π = 2 * Σ(occupied orbital energies)
        if n_filled > 0 and n_filled <= len(eigenvalues):
            total_pi_energy = 2 * np.sum(eigenvalues[:n_filled])
        else:
            total_pi_energy = 0.0
        
        # Delocalization energy = E_π - E_localized
        # E_localized ≈ n_double_bonds * 2β
        e_localized = n_pi_electrons  
        delocalization = total_pi_energy - e_localized
        
        # HOMO-LUMO gap
        if n_filled > 0 and n_filled < len(eigenvalues):
            homo = eigenvalues[n_filled - 1]
            lumo = eigenvalues[n_filled]
            gap = lumo - homo
        else:
            gap = 0.0
        
        features['huckel_delocalization'] = float(delocalization)
        features['huckel_homo_lumo_gap'] = float(gap)
        features['huckel_total_pi_energy'] = float(total_pi_energy)       
        features['huckel_lowest_orbital'] = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0
        features['huckel_highest_orbital'] = float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0
        features['huckel_orbital_spread'] = float(eigenvalues[-1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
        
    except:
        features['huckel_delocalization'] = 0.0
        features['huckel_homo_lumo_gap'] = 0.0
        features['huckel_total_pi_energy'] = 0.0
        features['huckel_lowest_orbital'] = 0.0
        features['huckel_highest_orbital'] = 0.0
        features['huckel_orbital_spread'] = 0.0
    
    return features

# =============================================================================
# 2. EXTENDED HÜCKEL / TIGHT-BINDING FEATURES
# =============================================================================

def compute_extended_huckel_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Extended Hückel Theory (EHT) features.
    
    EHT uses empirical VOIP values for diagonal elements:
    H_ii = VOIP_i (valence orbital ionization potential)
    
    And Wolfsberg-Helmholz formula for off-diagonal:
    H_ij = K * S_ij * (H_ii + H_jj) / 2
    where K ≈ 1.75 and S_ij is overlap integral
    """
    features = {}
    
    voip_sum = 0.0
    voip_2s_sum = 0.0
    voip_2p_sum = 0.0
    
    for elem, count in atom_counts.items():
        if elem == 'H':
            voip_sum += count * VOIP['H_1s']
        else:
            voip_sum += count * (VOIP.get(f'{elem}_2s', 0) + 3 * VOIP.get(f'{elem}_2p', 0))
            voip_2s_sum += count * VOIP.get(f'{elem}_2s', 0)
            voip_2p_sum += count * VOIP.get(f'{elem}_2p', 0)
    
    features['eht_voip_total'] = voip_sum * EV_TO_HARTREE
    features['eht_voip_2s'] = voip_2s_sum * EV_TO_HARTREE
    features['eht_voip_2p'] = voip_2p_sum * EV_TO_HARTREE
    
    slater_sum = 0.0
    for elem, count in atom_counts.items():
        if elem == 'H':
            slater_sum += count * SLATER_EXPONENTS['H_1s']
        else:
            slater_sum += count * SLATER_EXPONENTS.get(f'{elem}_2p', 1.5)

    features['eht_slater_sum'] = slater_sum
    
    n_heavy = sum(atom_counts.get(e, 0) for e in ['C', 'N', 'O', 'F'])
    if n_heavy > 0:
        features['eht_avg_slater'] = slater_sum / n_heavy
    else:
        features['eht_avg_slater'] = 0.0
    
    # Electronegativity-based self-energy χ = (I + A) / 2, η = (I - A) / 2
    chi_sum = 0.0
    eta_sum = 0.0
    for elem, count in atom_counts.items():
        props = ATOMIC_PROPS.get(elem, {})
        chi_sum += count * props.get('electronegativity', 2.5)
        eta_sum += count * props.get('hardness', 5.0)
    
    features['eht_chi_sum'] = chi_sum
    features['eht_eta_sum'] = eta_sum
    return features

# =============================================================================
# 3. ZPVE ESTIMATION
# =============================================================================

def compute_zpve_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Estimate Zero-Point Vibrational Energy (ZPVE).
    
    ZPVE = (1/2) * h * Σν_i
    
    For a molecule with N atoms:
    - Linear: 3N - 5 vibrational modes
    - Non-linear: 3N - 6 vibrational modes
    
    """
    features = {}
    
    n_atoms = sum(atom_counts.values())
    n_heavy = sum(atom_counts.get(e, 0) for e in ['C', 'N', 'O', 'F'])
    
    is_linear = (n_atoms <= 2) or (n_heavy == 2 and atom_counts.get('H', 0) <= 2)
    n_vib_modes = 3 * n_atoms - 5 if is_linear else 3 * n_atoms - 6
    
    features['zpve_n_modes'] = float(max(0, n_vib_modes))
    
    mol_h = Chem.AddHs(mol)
    
    freq_sum = 0.0
    mode_counts = defaultdict(int)
    
    for bond in mol_h.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        bt = bond.GetBondType()
        
        if 'H' in [a1, a2]:
            other = a1 if a2 == 'H' else a2
            if other == 'C':
                freq = BOND_FREQUENCIES['C-H_stretch']
                mode_counts['C-H_stretch'] += 1
            elif other == 'N':
                freq = BOND_FREQUENCIES['N-H_stretch']
                mode_counts['N-H_stretch'] += 1
            elif other == 'O':
                freq = BOND_FREQUENCIES['O-H_stretch']
                mode_counts['O-H_stretch'] += 1
            else:
                freq = 3000  
        else:
            if set([a1, a2]) == {'C'}:
                if bt == BondType.TRIPLE:
                    freq = BOND_FREQUENCIES['C≡C_stretch']
                    mode_counts['C≡C_stretch'] += 1
                elif bt == BondType.DOUBLE:
                    freq = BOND_FREQUENCIES['C=C_stretch']
                    mode_counts['C=C_stretch'] += 1
                else:
                    freq = BOND_FREQUENCIES['C-C_stretch']
                    mode_counts['C-C_stretch'] += 1
            elif set([a1, a2]) == {'C', 'N'}:
                if bt == BondType.TRIPLE:
                    freq = BOND_FREQUENCIES['C≡N_stretch']
                    mode_counts['C≡N_stretch'] += 1
                elif bt == BondType.DOUBLE:
                    freq = BOND_FREQUENCIES['C=N_stretch']
                    mode_counts['C=N_stretch'] += 1
                else:
                    freq = BOND_FREQUENCIES['C-N_stretch']
                    mode_counts['C-N_stretch'] += 1
            elif set([a1, a2]) == {'C', 'O'}:
                if bt == BondType.DOUBLE:
                    freq = BOND_FREQUENCIES['C=O_stretch']
                    mode_counts['C=O_stretch'] += 1
                else:
                    freq = BOND_FREQUENCIES['C-O_stretch']
                    mode_counts['C-O_stretch'] += 1
            elif set([a1, a2]) == {'C', 'F'}:
                freq = BOND_FREQUENCIES['C-F_stretch']
                mode_counts['C-F_stretch'] += 1
            elif set([a1, a2]) == {'N'}:
                if bt == BondType.DOUBLE:
                    freq = BOND_FREQUENCIES['N=N_stretch']
                    mode_counts['N=N_stretch'] += 1
                else:
                    freq = BOND_FREQUENCIES['N-N_stretch']
                    mode_counts['N-N_stretch'] += 1
            elif set([a1, a2]) == {'N', 'O'}:
                freq = BOND_FREQUENCIES['N-O_stretch']
                mode_counts['N-O_stretch'] += 1
            else:
                freq = 1000  # Default
        
        freq_sum += freq
    
  
    n_bends = max(0, n_vib_modes - len(mode_counts))
    avg_bend_freq = 800  # Average bending frequency
    freq_sum += n_bends * avg_bend_freq
    
    # ZPVE = (1/2) * h * Σν
    zpve_ha = 0.5 * freq_sum * CM_TO_HARTREE
    
    features['zpve_estimated'] = zpve_ha  # In Hartree
    features['zpve_freq_sum'] = freq_sum * CM_TO_HARTREE  
    
    high_freq_sum = (
        mode_counts.get('C-H_stretch', 0) * BOND_FREQUENCIES['C-H_stretch'] +
        mode_counts.get('N-H_stretch', 0) * BOND_FREQUENCIES['N-H_stretch'] +
        mode_counts.get('O-H_stretch', 0) * BOND_FREQUENCIES['O-H_stretch']
    )
    features['zpve_high_freq'] = 0.5 * high_freq_sum * CM_TO_HARTREE
    
    features['zpve_n_ch'] = float(mode_counts.get('C-H_stretch', 0))
    features['zpve_n_nh'] = float(mode_counts.get('N-H_stretch', 0))
    features['zpve_n_oh'] = float(mode_counts.get('O-H_stretch', 0))
    
    return features


# =============================================================================
# 4. SOLVATION FEATURES
# =============================================================================

def compute_solvation_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute solvation-related features.
    
    Key contributions to solvation free energy:
    1. Cavity formation (proportional to surface area)
    2. Electrostatic (Born model: q²/r * (1 - 1/ε))
    3. Dispersion (van der Waals interactions)
    
    G_solv ≈ G_cavity + G_electrostatic + G_dispersion
    """
    features = {}
    
    sasa_approx = 0.0
    for elem, count in atom_counts.items():
        r = VDW_RADII.get(elem, 1.5)
        # Approximate surface area contribution (spherical)
        sasa_approx += count * 4 * np.pi * r**2
    
    features['solv_sasa_approx'] = sasa_approx
    
    volume_approx = 0.0
    for elem, count in atom_counts.items():
        r = VDW_RADII.get(elem, 1.5)
        volume_approx += count * (4/3) * np.pi * r**3
    
    features['solv_volume_approx'] = volume_approx
    
    # G_cav ≈ γ * SASA
    gamma_water = SURFACE_TENSION['water']
    features['solv_cavity_water'] = gamma_water * sasa_approx
    
    total_polar = sum(atom_counts.get(e, 0) * ATOMIC_PROPS[e]['polarizability'] 
                      for e in atom_counts if e in ATOMIC_PROPS)
    features['solv_polarizability'] = total_polar
    
    try:
        en_sum = 0.0
        en_sq_sum = 0.0
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            en = ATOMIC_PROPS.get(sym, {}).get('electronegativity', 2.5)
            en_sum += en
            en_sq_sum += en**2
        
        n_atoms_mol = mol.GetNumAtoms()
        if n_atoms_mol > 0:
            en_mean = en_sum / n_atoms_mol
            en_var = en_sq_sum / n_atoms_mol - en_mean**2
            features['solv_en_variance'] = en_var
        else:
            features['solv_en_variance'] = 0.0
    except:
        features['solv_en_variance'] = 0.0
    
    try:
        tpsa = Descriptors.TPSA(mol)
        features['solv_tpsa'] = tpsa
    except:
        features['solv_tpsa'] = 0.0
    
    n_polar = atom_counts.get('N', 0) + atom_counts.get('O', 0) + atom_counts.get('F', 0)
    n_total = sum(atom_counts.values())
    features['solv_polar_fraction'] = n_polar / max(1, n_total)
    
    try:
        features['solv_hbd'] = float(rdMolDescriptors.CalcNumHBD(mol))
        features['solv_hba'] = float(rdMolDescriptors.CalcNumHBA(mol))
    except:
        features['solv_hbd'] = 0.0
        features['solv_hba'] = 0.0
    
    return features


def compute_solvation_correction(features: Dict[str, float], solvent: str = 'water') -> float:
    """
    Compute approximate solvation correction for a given solvent.
    
    Uses simplified GBSA-like model:
    G_solv ≈ -γ * SASA + electrostatic_term * (1 - 1/ε)
    
    Returns correction in Hartree.
    """
    epsilon = DIELECTRIC.get(solvent.lower(), 1.0)
    gamma = SURFACE_TENSION.get(solvent.lower(), 0.0)
    
    sasa = features.get('solv_sasa_approx', 0.0)
    
    g_cavity = gamma * sasa
    
    if epsilon > 1.0:
        polar_factor = features.get('solv_tpsa', 0.0) / 100.0  # Normalize
        g_elec = -0.001 * polar_factor * (1 - 1/epsilon)  # Very rough
    else:
        g_elec = 0.0
    
    return g_cavity + g_elec


# =============================================================================
# 5. U0 → SCF BRIDGE FEATURES
# =============================================================================

def compute_scf_bridge_features(mol, atom_counts: Dict[str, int], 
                                 zpve_features: Dict[str, float]) -> Dict[str, float]:
    """
    Features to bridge U0 (internal energy at 0K) to SCF energy.
    
    Key relationships:
    - U0 ≈ E_electronic + ZPVE
    - SCF = E_electronic (in DFT, this is the Kohn-Sham energy)
    - Therefore: SCF ≈ U0 - ZPVE
    
    Additional corrections:
    - Thermal corrections (U0 → U298, H298, G298)
    - Correlation energy differences between methods
    """
    features = {}
    
    zpve = zpve_features.get('zpve_estimated', 0.0)
    features['bridge_zpve'] = zpve
    
    n_atoms = sum(atom_counts.values())
    n_dof = 3 * n_atoms - 6  # Non-linear
    kT_298 = KB_HARTREE * 298.15
    thermal_corr = 0.5 * n_dof * kT_298
    features['bridge_thermal'] = thermal_corr
    features['bridge_u0_to_scf_corr'] = -zpve
    
    n_electrons = sum(atom_counts.get(e, 0) * ATOMIC_PROPS[e]['Z'] for e in atom_counts)
    features['bridge_n_electrons'] = float(n_electrons)
    
    features['bridge_corr_proxy'] = n_electrons ** 1.5
    
    return features


# =============================================================================
# 6. RING STRAIN 
# =============================================================================

BASE_RING_STRAIN = {3: 27.5, 4: 26.5, 5: 6.2, 6: 0.0, 7: 6.2, 8: 9.0}

def compute_ring_features(mol) -> Dict[str, float]:
    """Compute ring-related features."""
    features = {}
    ring_info = mol.GetRingInfo()
    rings = [set(r) for r in ring_info.AtomRings()]
    n_rings = len(rings)
    
    features['ring_count'] = float(n_rings)
    features['ring_aromatic'] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    features['ring_aliphatic'] = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    
    for size in [3, 4, 5, 6, 7]:
        features[f'ring_size_{size}'] = float(sum(1 for r in rings if len(r) == size))
    
    strain = 0.0
    for ring in rings:
        size = len(ring)
        strain += BASE_RING_STRAIN.get(size, 5.0)
    
    features['ring_strain_kcal'] = strain
    features['ring_strain_ha'] = strain * KCAL_TO_HARTREE
    
    n_fused = 0
    for i, r1 in enumerate(rings):
        for r2 in rings[i+1:]:
            if len(r1 & r2) >= 2:
                n_fused += 1
    
    features['ring_fused_count'] = float(n_fused)
    
    return features


# =============================================================================
# 7. FUNCTIONAL GROUPS
# =============================================================================

FG_PATTERNS = {
    'ketone': '[#6][CX3](=O)[#6]', 
    'aldehyde': '[CX3H1](=O)',
    'carboxyl': '[CX3](=O)[OX2H1]', 
    'ester': '[#6][CX3](=O)[OX2][#6]',
    'amide': '[NX3][CX3](=[OX1])', 
    'nitrile': '[NX1]#[CX2]',
    'amine_1': '[NX3;H2;!$(NC=O)]', 
    'amine_2': '[NX3;H1;!$(NC=O)]',
    'amine_3': '[NX3;H0;!$(NC=O)]', 
    'alcohol': '[OX2H][CX4]',
    'phenol': '[OX2H][cX3]', 
    'ether': '[OD2]([#6])[#6]',
    'nitro': '[N+](=O)[O-]',
    'fluorine': '[F]',
    'aromatic': 'c1ccccc1',
    'pyrrole': '[nH]1cccc1',
    'indole': 'c1ccc2[nH]ccc2c1',
    'imidazole': 'n1cc[nH]c1',
}

def get_functional_groups(mol) -> Dict[str, int]:
    """Count functional group occurrences."""
    fg_counts = {}
    for name, smarts in FG_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        fg_counts[name] = len(mol.GetSubstructMatches(pattern)) if pattern else 0
    fg_counts['amine'] = fg_counts['amine_1'] + fg_counts['amine_2'] + fg_counts['amine_3']
    return fg_counts


# =============================================================================
# 8. FINGERPRINT
# =============================================================================

def get_fingerprint(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if FP_GEN:
        fp = FP_GEN.GetFingerprint(mol)
        arr = np.zeros(FP_SIZE, dtype=np.float64)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        return arr
    return None


# =============================================================================
# MASTER EXTRACTION FUNCTION
# =============================================================================

def extract_all_features(smiles: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Extract all physics-enhanced features for a molecule.
    All energy values in HARTREE.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    atom_counts = count_atoms(smiles)
    if atom_counts is None:
        return None, None
    
    features = {}
    
    # Atom counts
    for e in ['H', 'C', 'N', 'O', 'F']:
        features[f'n_{e}'] = float(atom_counts[e])
    
    n_heavy = sum(atom_counts.get(e, 0) for e in ['C', 'N', 'O', 'F'])
    n_total = sum(atom_counts.values())
    features['n_heavy'] = float(n_heavy)
    features['n_total'] = float(n_total)
    
    atomic_baseline = compute_atomic_baseline(atom_counts)
    features['atomic_baseline'] = atomic_baseline
    
    huckel = compute_huckel_features(mol)
    for k, v in huckel.items():
        features[k] = v
    
    eht = compute_extended_huckel_features(mol, atom_counts)
    for k, v in eht.items():
        features[k] = v
    
    zpve = compute_zpve_features(mol, atom_counts)
    for k, v in zpve.items():
        features[k] = v
    
    solv = compute_solvation_features(mol, atom_counts)
    for k, v in solv.items():
        features[k] = v
    
    bridge = compute_scf_bridge_features(mol, atom_counts, zpve)
    for k, v in bridge.items():
        features[k] = v
    
    ring = compute_ring_features(mol)
    for k, v in ring.items():
        features[k] = v
    
    fg_counts = get_functional_groups(mol)
    for fg, cnt in fg_counts.items():
        features[f'fg_{fg}'] = float(cnt)
    
    features['n_bonds'] = float(mol.GetNumBonds())
    features['n_rotatable'] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    features['frac_sp3'] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    
    for k in features:
        if not np.isfinite(features[k]):
            features[k] = 0.0
    
    metadata = {
        'smiles': smiles,
        'atomic_baseline': atomic_baseline,
        'n_heavy': n_heavy,
        'n_atoms': n_total,
        'atom_counts': atom_counts,
        'fg_counts': fg_counts,
        'zpve_estimated': zpve.get('zpve_estimated', 0.0),
    }
    
    return features, metadata


# =============================================================================
# TESTING
# =============================================================================

print("\n" + "-" * 75)
print("Testing Physics-Enhanced Feature Extraction")
print("-" * 75)

test_mols = [
    ("C", "methane"),
    ("CCO", "ethanol"),
    ("c1ccccc1", "benzene"),
    ("c1ccc2[nH]ccc2c1", "indole"),
    ("Oc1ccccc1", "phenol"), 
    ("n1cc[nH]c1", "imidazole"), 
    ("c1ccncc1", "pyridine"),
    ("COc1ccccc1", "anisole"),
    ("Nc1ccccc1", "aniline"),
    ("COc1ccc(N)cc1", "p-anisidine"),
    ("COc1ccccc1N", "o-anisidine"),
    ("COc1cc(N)cc(OC)c1", "3,5-dimethoxyaniline"),
    ("NC(Cc1c[nH]c2ccccc12)C(=O)O", "tryptophan"),
    ("NC(CC(=O)O)C(=O)O", "aspartic_acid"),      
    ("NC(CCC(=O)O)C(=O)O", "glutamic_acid"),     
    ("NC(CO)C(=O)O", "serine"),                  
    ("CC(O)C(N)C(=O)O", "threonine"),          
    ("NC(Cc1ccc(O)cc1)C(=O)O", "tyrosine"),      
    ("NC(CCCNC(N)=N)C(=O)O", "arginine"),         
    ("NCCCCC(N)C(=O)O", "lysine"),                 
    ("NC(Cc1c[nH]cn1)C(=O)O", "histidine"),        
    ("NC(Cc1c[nH]c2ccccc12)C(=O)O", "tryptophan"), 
    ("NC(CCC(N)=O)C(=O)O", "glutamine"),           
    ("NC(CC(N)=O)C(=O)O", "asparagine"),           
    ("NC(Cc1ccccc1)C(=O)O", "phenylalanine"),
    ("NC(Cc1ccc(O)cc1)C(=O)O", "tyrosine"),
    ("NC(Cc1c[nH]cn1)C(=O)O", "histidine")
]

for smi, name in test_mols:
    f, m = extract_all_features(smi)
    if f:
        print(f"\n{name} ({smi}):")
        print(f"    N_heavy: {m['n_heavy']}")
        print(f"    Hückel π-atoms: {f['huckel_n_pi_atoms']:.0f}")
        print(f"    Hückel delocalization: {f['huckel_delocalization']:.4f} β")
        print(f"    ZPVE estimated: {f['zpve_estimated']:.6f} Ha ({f['zpve_estimated']*HARTREE_TO_KCAL:.2f} kcal/mol)")
        print(f"    Bridge U0→SCF: {f['bridge_u0_to_scf_corr']:.6f} Ha")

sample_f, _ = extract_all_features("CCO")
print(f"\nTotal features: {len(sample_f)}")

print("\n" + "=" * 75)
print("Feature Engineering Complete")
print("=" * 75)
sys.stdout.flush()


#---Training begins----


print("=" * 75)
print("Pysics Enhanced Training")
print("=" * 75)
sys.stdout.flush()

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_SIZE = 50000
RANDOM_STATE = 42
HARTREE_TO_KCAL = 627.509474
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL

print(f"\nConfiguration:")
print(f"    Training size: {TRAINING_SIZE:,}")
print(f"    All calculations in HARTREE")

# =============================================================================
# PREREQUISITES
# =============================================================================

print("\n" + "-" * 75)
print("Checking prerequisites...")

required = ['qm9_df', 'extract_all_features', 'get_fingerprint', 'FP_SIZE', 'ATOMIC_ENERGIES_B3LYP']
for r in required:
    if r not in globals():
        raise RuntimeError(f"Missing: {r}. Run Cells 1-2 first.")

print("    All found ✓")
print(f"    QM9 size: {len(qm9_df):,}")

has_zpve = 'zpve' in qm9_df.columns
print(f"    QM9 has ZPVE column: {has_zpve}")

sys.stdout.flush()

# =============================================================================
# PHASE 1: DATA PREPARATION
# =============================================================================

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

# ----------------- SAMPLE WEIGHTS (focus aromatic + >9 HA) -------------------
ring_arom_arr = np.array([fd.get('ring_aromatic', 0.0) for fd in feature_dicts], dtype=float)
w = np.ones(len(X))
w[(n_heavy >= 7) & (n_heavy <= 9)] *= 1.35
w[(n_heavy > 9)] *= 1.15
w *= (1.0 + 0.30 * np.clip(ring_arom_arr, 0, 2))
w *= len(w) / w.sum()
# -----------------------------------------------------------------------------

sys.stdout.flush()

# =============================================================================
# PHASE 2: DETAILED ZPVE ANALYSIS
# =============================================================================

print("\n" + "=" * 75)
print("PHASE 2: Detailed ZPVE Analysis")
print("=" * 75)

print("\n    ZPVE by total atoms:")
print(f"    {'N_atoms':<10} {'Count':<8} {'Mean ZPVE':<12} {'Per atom':<12} {'Per H':<12}")
print("    " + "-" * 54)

zpve_data = []
for n in sorted(set(n_atoms)):
    mask = n_atoms == n
    count = np.sum(mask)
    if count >= 30:
        mean_zpve = np.mean(y_zpve[mask])
        mean_nH = np.mean(n_H[mask])
        zpve_per_atom = mean_zpve / n
        zpve_per_H = mean_zpve / mean_nH if mean_nH > 0 else 0
        zpve_data.append({
            'n_atoms': n,
            'count': count,
            'mean_zpve': mean_zpve,
            'mean_nH': mean_nH,
            'zpve_per_atom': zpve_per_atom,
        })
        print(f"    {n:<10} {count:<8} {mean_zpve:<12.6f} {zpve_per_atom:<12.6f} {zpve_per_H:<12.6f}")

# ZPVE = a * N_atoms + b * N_H + c * N_atoms² + d
print("\n    Fitting multi-component ZPVE model...")

X_zpve = np.column_stack([
    n_atoms,                    # Linear in total atoms
    n_H,                        # H atoms (high-frequency C-H stretches)
    n_atoms ** 2,               # Quadratic (more modes = superlinear)
    n_bonds,                    # More bonds = more stretching modes
    np.ones(len(n_atoms)),      # Intercept
])

# Fit with Ridge regression
from sklearn.linear_model import Ridge as RidgeReg
zpve_linear_model = RidgeReg(alpha=0.1)
zpve_linear_model.fit(X_zpve, y_zpve, sample_weight=w)

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

print("\n    Extrapolation estimates:")
test_cases = [
    (20, 12, 20),   
    (30, 18, 32),   
    (35, 22, 38),  
    (40, 26, 44),   
]
for n_at, n_h, n_bd in test_cases:
    zpve_est = (ZPVE_COEFFS['n_atoms'] * n_at + 
                ZPVE_COEFFS['n_H'] * n_h + 
                ZPVE_COEFFS['n_atoms_sq'] * n_at**2 +
                ZPVE_COEFFS['n_bonds'] * n_bd +
                ZPVE_COEFFS['intercept'])
    print(f"        N_atoms={n_at}, N_H={n_h}: ZPVE ≈ {zpve_est:.4f} Ha ({zpve_est * HARTREE_TO_KCAL:.1f} kcal/mol)")

sys.stdout.flush()

# =============================================================================
# PHASE 3: ML ZPVE RESIDUAL MODEL
# =============================================================================

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
zpve_residual_model.fit(X_scaled, zpve_residual, sample_weight=w)

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

# =============================================================================
# PHASE 4: FORMATION ENERGY MODEL
# =============================================================================

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
huber.fit(X_ml, y_formation, sample_weight=w)
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
ridge.fit(X_enhanced, y_formation, sample_weight=w)
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
gb_main.fit(X_enhanced, res_ridge, sample_weight=w)
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
gb_refine.fit(X_enhanced, res_gb, sample_weight=w)
pred_refine = gb_refine.predict(X_enhanced)

pred_formation_ml = pred_ridge + pred_gb + pred_refine
sys.stdout.flush()

# =============================================================================
# PHASE 5: SIZE, π, AND FG CORRECTIONS (REPLACED)
# =============================================================================

print("\n" + "=" * 75)
print("PHASE 5: Size, π, and FG Corrections")
print("=" * 75)

res_for_corr = y_formation - pred_formation_ml

print("\n[Size Correction with isotonic trend + quadratic core + linear tail]")

res_for_corr = y_formation - pred_formation_ml

size_residuals = defaultdict(list)
for i, n in enumerate(n_heavy):
    size_residuals[int(n)].append(res_for_corr[i])

sizes = sorted([s for s in size_residuals.keys() if len(size_residuals[s]) >= 50])
means = [float(np.mean(size_residuals[s])) for s in sizes]

if len(sizes) >= 3:
    sizes_arr = np.asarray(sizes, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    weights_arr = np.asarray([len(size_residuals[s]) for s in sizes], dtype=float)

    # decide monotonic direction (no 'auto' to avoid version issues)
    try:
        corr = np.corrcoef(sizes_arr, means_arr)[0, 1]
        inc = bool(np.nan_to_num(corr) >= 0.0)
    except Exception:
        inc = True

    try:
        iso = IsotonicRegression(increasing=inc, out_of_bounds='clip')
        iso.fit(sizes_arr, means_arr, sample_weight=weights_arr)
        trend = iso.predict(sizes_arr)
    except Exception:
        trend = means_arr

    SIZE_COEFFS = np.polyfit(sizes_arr, trend, 2)
    # define piecewise-linear tail beyond observed max size
    size_max = int(np.max(sizes_arr))
    a, b, c = SIZE_COEFFS  # quadratic coeffs
    slope = float(2.0 * a * size_max + b)
else:
    SIZE_COEFFS = [0.0, 0.0, 0.0]
    size_max = 9
    slope = 0.0

# apply piecewise correction (quadratic core up to size_max, linear tail after)
size_core = np.polyval(SIZE_COEFFS, np.minimum(n_heavy, size_max))
size_tail = np.maximum(0, n_heavy - size_max) * slope
size_corr = size_core + size_tail

print("\n[π-System Correction (aromatic rings + Hückel delocalization)]")
pi_X = np.column_stack([
    ring_arom_arr,
    np.array([fd.get('huckel_delocalization', 0.0) for fd in feature_dicts], dtype=float),
])
pi_ridge = Ridge(alpha=0.01, random_state=RANDOM_STATE)
pi_ridge.fit(pi_X, res_for_corr - size_corr, sample_weight=w)
pi_corr = pi_ridge.predict(pi_X)

print("\n[FG Corrections]")
res_after_size_pi = res_for_corr - size_corr - pi_corr

fg_residuals = defaultdict(list)
for i, meta in enumerate(metadata_list):
    for fg, count in meta['fg_counts'].items():
        if count > 0:
            fg_residuals[fg].append(res_after_size_pi[i])

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

# =============================================================================
# PHASE 6: FINAL EVALUATION
# =============================================================================

print("\n" + "=" * 75)
print("PHASE 6: Final Evaluation")
print("=" * 75)

pred_formation_total = pred_formation_ml + size_corr + pi_corr + fg_corr
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

# =============================================================================
# PHASE 7: STORE MODEL
# =============================================================================

print("\n" + "=" * 75)
print("PHASE 7: Storing Model")
print("=" * 75)

MODEL_COMPONENTS = {
    'version': 'aggressive_zpve',
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
    'size_max': int(size_max),
    'size_tail': float(slope),
    'pi_intercept': float(pi_ridge.intercept_),
    'pi_coef_ring': float(pi_ridge.coef_[0]),
    'pi_coef_deloc': float(pi_ridge.coef_[1]),
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
}

feature_names_global = feature_names

print(f"\n    Model stored!")
print(f"    ZPVE formula: {ZPVE_COEFFS['n_atoms']:.5f}×N + {ZPVE_COEFFS['n_H']:.5f}×H + {ZPVE_COEFFS['n_atoms_sq']:.7f}×N² + ...")

print("\n" + "=" * 75)
print("Training Complete. Moving to Scaling...")
print("=" * 75)
sys.stdout.flush()

# -------------------------------------------------------------------

print("=" * 75)
print("Aggressive ZPVE Scaling")
print("=" * 75)
sys.stdout.flush()

# =============================================================================
# PREREQUISITES
# =============================================================================

print("\nChecking prerequisites...")

required = ['MODEL_COMPONENTS', 'MODEL_PARAMS', 'extract_all_features', 
            'get_fingerprint', 'FP_SIZE']
for r in required:
    if r not in globals():
        raise RuntimeError(f"Missing: {r}. Run Cells 1-3 first.")

if 'feature_names_global' in globals():
    feature_names = feature_names_global
elif 'feature_names' in MODEL_PARAMS:
    feature_names = MODEL_PARAMS['feature_names']
else:
    raise RuntimeError("feature_names not found")

if 'zpve_coeffs' not in MODEL_COMPONENTS:
    print("    WARNING: Using default ZPVE coefficients")
    MODEL_COMPONENTS['zpve_coeffs'] = {
        'n_atoms': 0.004,
        'n_H': 0.002,
        'n_atoms_sq': 0.00005,
        'n_bonds': 0.001,
        'intercept': 0.01,
    }

print("    All prerequisites found ✓")

ZPVE_COEFFS = MODEL_COMPONENTS['zpve_coeffs']
print(f"    ZPVE coefficients loaded")

# =============================================================================
# CONSTANTS
# =============================================================================

HARTREE_TO_KCAL = 627.509474
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL

DIELECTRIC = {
    'vacuum': 1.0, 'gas': 1.0,
    'hexane': 1.88, 'benzene': 2.27, 'toluene': 2.38,
    'chloroform': 4.81, 'thf': 7.58,
    'dichloromethane': 8.93, 'dcm': 8.93,
    'acetone': 20.7, 'ethanol': 24.3, 'methanol': 32.7,
    'acetonitrile': 37.5, 'dmso': 46.7, 'water': 78.4,
}

# =============================================================================
# ZPVE PREDICTION
# =============================================================================

def predict_zpve(n_atoms: int, n_H: int, n_bonds: int, 
                 features: Dict, X_scaled: np.ndarray) -> float:
    """
    Predict ZPVE using multi-component model.
    
    ZPVE = a×N_atoms + b×N_H + c×N_atoms² + d×N_bonds + intercept + ML_residual
    
    This scales properly for large molecules because:
    - N_H term captures high-frequency X-H stretches
    - N_atoms² term captures the superlinear growth of low-frequency modes
    - N_bonds captures stretching mode contributions
    """
    coeffs = MODEL_COMPONENTS['zpve_coeffs']
    
    zpve_baseline = (
        coeffs['n_atoms'] * n_atoms +
        coeffs['n_H'] * n_H +
        coeffs['n_atoms_sq'] * (n_atoms ** 2) +
        coeffs['n_bonds'] * n_bonds +
        coeffs['intercept']
    )
    
    if 'zpve_residual_model' in MODEL_COMPONENTS:
        try:
            residual = MODEL_COMPONENTS['zpve_residual_model'].predict(X_scaled)[0]
            
            if n_atoms > 25:
                damping = max(0.0, 1.0 - 0.05 * (n_atoms - 25))
                residual *= damping
            
            zpve_total = zpve_baseline + residual
        except:
            zpve_total = zpve_baseline
    else:
        zpve_total = zpve_baseline
    
    return max(0.02, zpve_total)


# =============================================================================
# SOLVATION
# =============================================================================

def compute_solvation_correction(features: Dict, metadata: Dict, 
                                  solvent: str = 'vacuum') -> float:
    """Compute solvation free energy (NEGATIVE = stabilization)."""
    solvent = solvent.lower()
    
    if solvent in ['vacuum', 'gas', 'none', '']:
        return 0.0
    
    epsilon = DIELECTRIC.get(solvent, 1.0)
    if epsilon <= 1.0:
        return 0.0
    
    tpsa = features.get('solv_tpsa', 0.0)
    hbd = features.get('solv_hbd', 0.0)
    hba = features.get('solv_hba', 0.0)
    n_N = features.get('n_N', 0)
    n_O = features.get('n_O', 0)
    sasa = features.get('solv_sasa_approx', 100.0)
    polarizability = features.get('solv_polarizability', 5.0)
    n_heavy = features.get('n_heavy', 5)
    
    polarity = min(2.0, (tpsa / 150.0) + 0.15 * (hbd + hba) + 0.05 * (n_N + n_O))
    kirkwood = (epsilon - 1) / (2 * epsilon + 1)
    g_elec = -12.0 * polarity * kirkwood
    
    gamma = 0.007 if solvent == 'water' else 0.004
    g_cav = gamma * sasa
    
    disp_factor = 0.015 if solvent == 'water' else 0.022
    g_disp = -disp_factor * (polarizability + 0.5 * n_heavy)
    
    if solvent == 'water':
        g_hbond = -1.5 * hbd - 0.8 * hba
    elif solvent in ['methanol', 'ethanol']:
        g_hbond = -1.0 * hbd - 0.5 * hba
    else:
        g_hbond = 0.0
    
    g_total_kcal = g_elec + g_cav + g_disp + g_hbond
    return g_total_kcal * KCAL_TO_HARTREE


# =============================================================================
# MAIN PREDICTION
# =============================================================================

def predict_molecule(
    smiles: str,
    solvent: str = 'vacuum',
    verbose: bool = False,
    return_breakdown: bool = False
) -> Optional[Dict]:
    """
    Predict SCF energy for a molecule.
    
    SCF = atomic_baseline + formation - ZPVE + solvation
    """
    features, metadata = extract_all_features(smiles)
    if features is None:
        return None
    
    X = np.array([[features.get(fn, 0.0) for fn in feature_names]])
    X_scaled = MODEL_COMPONENTS['scaler_main'].transform(X)
    X_scaled = np.clip(np.nan_to_num(X_scaled), -10, 10)
    
    fp = get_fingerprint(smiles)
    if fp is None:
        fp = np.zeros(FP_SIZE)
    
    X_full = np.hstack([X_scaled, fp.reshape(1, -1)])
    X_ml = MODEL_COMPONENTS['scaler_ml'].transform(X_full)
    X_ml = np.clip(np.nan_to_num(X_ml), -10, 10)
    
    X_top = X_ml[:, MODEL_COMPONENTS['top_idx']]
    X_poly = MODEL_COMPONENTS['poly'].transform(X_top)
    X_enhanced = np.hstack([X_ml, X_poly])
    
    pred_ridge = MODEL_COMPONENTS['ridge'].predict(X_enhanced)[0]
    pred_gb = MODEL_COMPONENTS['gb_main'].predict(X_enhanced)[0]
    pred_refine = MODEL_COMPONENTS['gb_refine'].predict(X_enhanced)[0]
    formation_ml = pred_ridge + pred_gb + pred_refine
    
    n_heavy = metadata['n_heavy']
    n_atoms = metadata['n_atoms']
    n_H = metadata['atom_counts'].get('H', 0)
    n_bonds = int(features.get('n_bonds', n_atoms - 1))
    
    coeffs = MODEL_COMPONENTS['size_coeffs']
    nmax = MODEL_COMPONENTS.get('size_max', 9)
    slope = MODEL_COMPONENTS.get('size_tail', 0.0)
    base = np.polyval(coeffs, min(n_heavy, nmax))
    tail = (n_heavy - nmax) * slope if n_heavy > nmax else 0.0
    size_corr = base + tail

    ring = float(features.get('ring_aromatic', 0.0))
    deloc = float(features.get('huckel_delocalization', 0.0))
    pi_corr = (MODEL_COMPONENTS.get('pi_intercept', 0.0) +
               MODEL_COMPONENTS.get('pi_coef_ring', 0.0) * ring +
               MODEL_COMPONENTS.get('pi_coef_deloc', 0.0) * deloc)
    
    fg_corr = 0.0
    for fg, count in metadata['fg_counts'].items():
        if fg in MODEL_COMPONENTS['fg_corrections'] and count > 0:
            fg_corr += MODEL_COMPONENTS['fg_corrections'][fg]
    
    formation_total = formation_ml + size_corr + pi_corr + fg_corr
    
    atomic_baseline = metadata['atomic_baseline']
    u0 = atomic_baseline + formation_total
    
    zpve = predict_zpve(n_atoms, n_H, n_bonds, features, X_scaled)
    
    e_elec = u0 - zpve
    
    solv_corr = compute_solvation_correction(features, metadata, solvent)
    
    scf_gas = e_elec
    scf_solvated = e_elec + solv_corr
    
    base_unc = MODEL_PARAMS.get('u0_mae', 0.005)
    zpve_unc = MODEL_PARAMS.get('zpve_mae', 0.003)
    
    extrap_factor = 1.0 + 0.15 * max(0, n_heavy - 9)
    solv_unc = 0.008 if solvent not in ['vacuum', 'gas', ''] else 0.0
    
    total_unc = np.sqrt(base_unc**2 + zpve_unc**2 + solv_unc**2) * extrap_factor
    
    result = {
        'u0': u0,
        'zpve': zpve,
        'e_elec': e_elec,
        'scf_gas': scf_gas,
        'solvation': solv_corr,
        'scf_solvated': scf_solvated,
        'uncertainty': total_unc,
        
        'u0_kcal': u0 * HARTREE_TO_KCAL,
        'zpve_kcal': zpve * HARTREE_TO_KCAL,
        'e_elec_kcal': e_elec * HARTREE_TO_KCAL,
        'scf_gas_kcal': scf_gas * HARTREE_TO_KCAL,
        'solvation_kcal': solv_corr * HARTREE_TO_KCAL,
        'scf_solvated_kcal': scf_solvated * HARTREE_TO_KCAL,
        'uncertainty_kcal': total_unc * HARTREE_TO_KCAL,
        
        'smiles': smiles,
        'n_heavy': n_heavy,
        'n_atoms': n_atoms,
        'n_H': n_H,
        'n_bonds': n_bonds,
        'atomic_baseline': atomic_baseline,
        'solvent': solvent,
        'extrapolated': n_heavy > 9,
    }
    
    if return_breakdown:
        coeffs_z = MODEL_COMPONENTS['zpve_coeffs']
        zpve_baseline = (coeffs_z['n_atoms'] * n_atoms + coeffs_z['n_H'] * n_H + 
                         coeffs_z['n_atoms_sq'] * n_atoms**2 + coeffs_z['n_bonds'] * n_bonds +
                         coeffs_z['intercept'])
        
        result['breakdown'] = {
            'atomic_baseline': atomic_baseline,
            'formation_ml': formation_ml,
            'size_corr': size_corr,
            'pi_corr': pi_corr,
            'fg_corr': fg_corr,
            'formation_total': formation_total,
            'zpve_baseline': zpve_baseline,
            'zpve_final': zpve,
        }
    
    if verbose:
        ext = " [EXTRAPOLATED]" if result['extrapolated'] else ""
        print(f"\n{'='*75}")
        print(f"PREDICTION: {smiles}{ext}")
        print(f"{'='*75}")
        print(f"    N_heavy: {n_heavy}, N_atoms: {n_atoms}, N_H: {n_H}, Solvent: {solvent}")
        
        print(f"\n    ENERGY CALCULATION (Hartree):")
        print(f"        Atomic baseline:    {atomic_baseline:18.8f}")
        print(f"      + Formation:          {formation_total:+18.8f}")
        print(f"        ─────────────────────────────────────────────")
        print(f"      = U0:                 {u0:18.8f}")
        print(f"      - ZPVE:               {zpve:18.8f}")
        print(f"        ─────────────────────────────────────────────")
        print(f"      = E_elec (gas):       {e_elec:18.8f}")
        print(f"      + Solvation:          {solv_corr:+18.8f}")
        print(f"        ─────────────────────────────────────────────")
        print(f"      = SCF (predicted):    {scf_solvated:18.8f}")
        print(f"        Uncertainty:        ±{total_unc:17.8f}")
        
        print(f"\n    IN kcal/mol:")
        print(f"        U0:                 {u0 * HARTREE_TO_KCAL:18.2f}")
        print(f"        ZPVE:               {zpve * HARTREE_TO_KCAL:18.2f}")
        print(f"        E_elec:             {e_elec * HARTREE_TO_KCAL:18.2f}")
        print(f"        Solvation:          {solv_corr * HARTREE_TO_KCAL:+18.2f}")
        print(f"        SCF:                {scf_solvated * HARTREE_TO_KCAL:18.2f}")
        
        print(f"\n    ZPVE BREAKDOWN:")
        coeffs_z = MODEL_COMPONENTS['zpve_coeffs']
        zpve_from_atoms = coeffs_z['n_atoms'] * n_atoms
        zpve_from_H = coeffs_z['n_H'] * n_H
        zpve_from_sq = coeffs_z['n_atoms_sq'] * n_atoms**2
        zpve_from_bonds = coeffs_z['n_bonds'] * n_bonds
        print(f"        From N_atoms ({n_atoms}):  {zpve_from_atoms:.6f} Ha")
        print(f"        From N_H ({n_H}):       {zpve_from_H:.6f} Ha")
        print(f"        From N²:            {zpve_from_sq:.6f} Ha")
        print(f"        From bonds ({n_bonds}):   {zpve_from_bonds:.6f} Ha")
        print(f"        Intercept:          {coeffs_z['intercept']:.6f} Ha")
        print(f"        Total ZPVE:         {zpve:.6f} Ha ({zpve * HARTREE_TO_KCAL:.2f} kcal/mol)")
        
        print(f"{'='*75}")
    
    return result


def predict_batch(smiles_list: List[str], solvent: str = 'vacuum',
                  show_progress: bool = True) -> List[Optional[Dict]]:
    """Batch prediction."""
    results = []
    for i, smi in enumerate(smiles_list):
        if show_progress and i % 50 == 0 and i > 0:
            print(f"    {i}/{len(smiles_list)}")
        try:
            results.append(predict_molecule(smi, solvent=solvent))
        except:
            results.append(None)
    return results


def check_within_range(pred_scf: float, scf_min: float, scf_max: float) -> bool:
    """Check if prediction is within experimental SCF range."""
    return scf_min <= pred_scf <= scf_max


# =============================================================================
# TEST
# =============================================================================

print("\n" + "-" * 75)
print("Testing ZPVE Scaling")
print("-" * 75)

test_cases = [
    ("C", "methane"),
    ("CCO", "ethanol"),
    ("c1ccccc1", "benzene"),
    ("c1ccc2[nH]ccc2c1", "indole"),
    ("Oc1ccccc1", "phenol"), 
    ("n1cc[nH]c1", "imidazole"), 
    ("c1ccncc1", "pyridine"),
    ("COc1ccccc1", "anisole"),
    ("Nc1ccccc1", "aniline"),
    ("COc1ccc(N)cc1", "p-anisidine"),
    ("COc1ccccc1N", "o-anisidine"),
    ("COc1cc(N)cc(OC)c1", "3,5-dimethoxyaniline"),
    ("NC(Cc1c[nH]c2ccccc12)C(=O)O", "tryptophan"),
    ("NC(CC(=O)O)C(=O)O", "aspartic_acid"),      
    ("NC(CCC(=O)O)C(=O)O", "glutamic_acid"),     
    ("NC(CO)C(=O)O", "serine"),                  
    ("CC(O)C(N)C(=O)O", "threonine"),          
    ("NC(Cc1ccc(O)cc1)C(=O)O", "tyrosine"),      
    ("NC(CCCNC(N)=N)C(=O)O", "arginine"),         
    ("NCCCCC(N)C(=O)O", "lysine"),                 
    ("NC(Cc1c[nH]cn1)C(=O)O", "histidine"),        
    ("NC(Cc1c[nH]c2ccccc12)C(=O)O", "tryptophan"), 
    ("NC(CCC(N)=O)C(=O)O", "glutamine"),           
    ("NC(CC(N)=O)C(=O)O", "asparagine"),           
    ("NC(Cc1ccccc1)C(=O)O", "phenylalanine"),
    ("NC(Cc1ccc(O)cc1)C(=O)O", "tyrosine"),
    ("NC(Cc1c[nH]cn1)C(=O)O", "histidine")

]

print(f"\n{'Molecule':<12} {'N_atoms':<8} {'N_H':<6} {'ZPVE (Ha)':<12} {'ZPVE (kcal)':<12}")
print("-" * 52)

for smi, name in test_cases:
    pred = predict_molecule(smi, verbose=False)
    if pred:
        print(f"{name:<12} {pred['n_atoms']:<8} {pred['n_H']:<6} {pred['zpve']:<12.6f} {pred['zpve_kcal']:<12.2f}")


predict_molecule("CN(C)CCc1c[nH]c2ccccc12", solvent='water', verbose=True)

print("\n" + "=" * 75)
print("Training Complete. Systerm Ready for Input")

sys.stdout.flush()
