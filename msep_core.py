"""
MSEP Core Library
=================
Machine Learning SCF Energy Predictor

This module contains all the core functionality for MSEP:
- Physical constants and parameters
- Feature extraction functions
- Model loading and prediction functions

Usage:
    from msep_core import load_model, predict_molecule, predict_batch
    
    # Load the pre-trained model (do this once)
    load_model('msep_model.pkl')
    
    # Predict SCF energy for a molecule
    result = predict_molecule('CCO', solvent='water')
    print(f"SCF Energy: {result['scf_solvated']:.6f} Ha")
"""

import os
import sys
import pickle
import warnings
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, rdMolDescriptors, Crippen,
        rdchem, GetSymmSSSR, rdFingerprintGenerator
    )
    from rdkit.Chem.rdchem import HybridizationType, BondType
except ImportError:
    raise RuntimeError("RDKit is required. Install with: conda install -c conda-forge rdkit")

warnings.filterwarnings('ignore')

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.2114
BOHR_TO_ANGSTROM = 0.529177
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
KB_HARTREE = 3.1668e-6
CM_TO_HARTREE = 4.5563e-6

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

HUCKEL_ALPHA = {
    'C': 0.0,
    'N': 0.5,
    'O': 1.0,
    'F': 1.5,
    'S': 0.0,
}

HUCKEL_BETA = {
    ('C', 'C'): 1.0,
    ('C', 'N'): 1.0,
    ('C', 'O'): 0.8,
    ('N', 'N'): 0.9,
    ('N', 'O'): 0.8,
    ('O', 'O'): 0.7,
    ('C', 'F'): 0.7,
}

HETEROATOM_H = {
    'N_sp2': 0.5,
    'N_sp3': 1.5,
    'N_pyrrole': 1.5,
    'O_sp2': 1.0,
    'O_sp3': 2.0,
    'F': 3.0,
}

BOND_K = {
    ('C', 'N'): 1.0,
    ('C', 'O'): 0.8,
    ('C', 'F'): 0.7,
    ('N', 'N'): 0.9,
    ('N', 'O'): 0.8,
}

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
    'O_2p': -15.9,
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
# VIBRATIONAL FREQUENCY PARAMETERS
# =============================================================================

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

# =============================================================================
# SOLVATION PARAMETERS
# =============================================================================

DIELECTRIC = {
    'vacuum': 1.0, 'gas': 1.0,
    'hexane': 1.88, 'benzene': 2.27, 'toluene': 2.38,
    'chloroform': 4.81, 'thf': 7.58,
    'dichloromethane': 8.93, 'dcm': 8.93,
    'acetone': 20.7, 'ethanol': 24.3, 'methanol': 32.7,
    'acetonitrile': 37.5, 'dmso': 46.7, 'water': 78.4,
}

SURFACE_TENSION = {
    'vacuum': 0.0, 'gas': 0.0,
    'water': 0.00007,
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
# RING STRAIN PARAMETERS
# =============================================================================

BASE_RING_STRAIN = {3: 27.5, 4: 26.5, 5: 6.2, 6: 0.0, 7: 6.2, 8: 9.0}

# =============================================================================
# FUNCTIONAL GROUP PATTERNS
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

# =============================================================================
# FINGERPRINT SETTINGS
# =============================================================================

FP_SIZE = 256
FP_RADIUS = 2

try:
    FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
except:
    FP_GEN = None

# =============================================================================
# GLOBAL MODEL STORAGE
# =============================================================================

# These will be populated when load_model() is called
MODEL_COMPONENTS = {}
MODEL_PARAMS = {}
_feature_names = []
_model_loaded = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_hybridization(atom) -> str:
    """Get hybridization state of an atom."""
    hyb = atom.GetHybridization()
    if hyb == HybridizationType.SP3:
        return 'sp3'
    elif hyb == HybridizationType.SP2:
        return 'sp2'
    elif hyb == HybridizationType.SP:
        return 'sp'
    return 'sp3'


def get_bond_order(bond) -> float:
    """Get bond order from RDKit bond."""
    bt = bond.GetBondType()
    if bt == BondType.SINGLE:
        return 1.0
    elif bt == BondType.DOUBLE:
        return 2.0
    elif bt == BondType.TRIPLE:
        return 3.0
    elif bt == BondType.AROMATIC:
        return 1.5
    return 1.0


def count_atoms(smiles: str) -> Optional[Dict[str, int]]:
    """Count atoms by element in a SMILES string."""
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
            return None  # Unsupported element
    return counts


def compute_atomic_baseline(atom_counts: Dict[str, int]) -> float:
    """Compute atomic baseline energy in Hartree."""
    return sum(atom_counts[elem] * ATOMIC_ENERGIES_B3LYP[elem] for elem in atom_counts)


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string is valid."""
    if smiles is None or (hasattr(smiles, '__len__') and len(str(smiles).strip()) == 0):
        return False
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================

def compute_huckel_features(mol) -> Dict[str, float]:
    """Compute Hückel theory-based features for π-systems."""
    features = {}
    
    pi_atoms = []
    pi_atom_idx = []
    
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        hyb = get_hybridization(atom)
        is_arom = atom.GetIsAromatic()
        
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
        features['huckel_lowest_orbital'] = 0.0
        features['huckel_highest_orbital'] = 0.0
        features['huckel_orbital_spread'] = 0.0
        return features
    
    H = np.zeros((n_pi, n_pi))
    
    alpha_sum = 0.0
    for i, pa in enumerate(pi_atoms):
        sym = pa['symbol']
        h_i = HUCKEL_ALPHA.get(sym, 0.0)
        H[i, i] = h_i
        alpha_sum += h_i
    
    features['huckel_alpha_sum'] = alpha_sum
    
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
            
            key = tuple(sorted([s1, s2]))
            k_ij = HUCKEL_BETA.get(key, 1.0)
            
            H[i, j] = k_ij
            H[j, i] = k_ij
            beta_sum += k_ij
    
    features['huckel_beta_sum'] = beta_sum
    
    try:
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.sort(eigenvalues)
        
        n_pi_electrons = n_pi
        n_filled = n_pi_electrons // 2
        
        if n_filled > 0 and n_filled <= len(eigenvalues):
            total_pi_energy = 2 * np.sum(eigenvalues[:n_filled])
        else:
            total_pi_energy = 0.0
        
        e_localized = n_pi_electrons
        delocalization = total_pi_energy - e_localized
        
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


def compute_extended_huckel_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """Compute Extended Hückel Theory (EHT) features."""
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
    
    chi_sum = 0.0
    eta_sum = 0.0
    for elem, count in atom_counts.items():
        props = ATOMIC_PROPS.get(elem, {})
        chi_sum += count * props.get('electronegativity', 2.5)
        eta_sum += count * props.get('hardness', 5.0)
    
    features['eht_chi_sum'] = chi_sum
    features['eht_eta_sum'] = eta_sum
    return features

def compute_lone_pair_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute features related to lone pair interactions.
    
    Physics basis:
    - Lone pairs contribute to total energy through repulsion and hyperconjugation
    - Oxygen lone pairs are particularly important due to high electronegativity
    - Adjacent lone pairs have repulsive interactions
    """
    features = {}
    
    # Count lone pairs by element and hybridization
    lp_counts = {'N_sp3': 0, 'N_sp2': 0, 'O_sp3': 0, 'O_sp2': 0, 'F': 0}
    lp_atoms = []  
    
    mol_h = Chem.AddHs(mol)
    
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ['N', 'O', 'F']:
            continue
            
        hyb = get_hybridization(atom)
        n_bonds = atom.GetDegree()
        
        # Calculate lone pairs: (valence electrons - bonding electrons) / 2
        if sym == 'N':
            n_lp = 1 if hyb == 'sp3' else (1 if hyb == 'sp2' and not atom.GetIsAromatic() else 0)
            key = f'N_{hyb}'
        elif sym == 'O':
            n_lp = 2  # O always has 2 lone pairs (sp3) or 2 (sp2, e.g., carbonyl)
            key = f'O_{hyb}'
        elif sym == 'F':
            n_lp = 3
            key = 'F'
        else:
            continue
            
        if key in lp_counts:
            lp_counts[key] += 1
        lp_atoms.append((atom.GetIdx(), n_lp, sym))
    
    # Basic lone pair counts
    total_lp = sum(v for v in lp_counts.values())
    features['lp_total_count'] = float(total_lp)
    features['lp_N_sp3'] = float(lp_counts['N_sp3'])
    features['lp_N_sp2'] = float(lp_counts['N_sp2'])
    features['lp_O_sp3'] = float(lp_counts['O_sp3'])
    features['lp_O_sp2'] = float(lp_counts['O_sp2'])
    features['lp_F'] = float(lp_counts['F'])
    
    # Lone pair energy contribution (empirical, based on orbital energies)
    # Oxygen lone pairs are lower in energy (more stable) than nitrogen
    LP_ORBITAL_ENERGIES = {
        'N_sp3': -0.52,  
        'N_sp2': -0.48,
        'O_sp3': -0.58,
        'O_sp2': -0.62,  
        'F': -0.70,
    }
    
    lp_energy = sum(lp_counts[k] * LP_ORBITAL_ENERGIES[k] * 2  # 2 electrons per lone pair
                    for k in lp_counts)
    features['lp_orbital_energy'] = lp_energy
    
    # Adjacent lone pair repulsion (1-2 and 1-3 interactions)
    lp_repulsion = 0.0
    lp_idx_map = {item[0]: item for item in lp_atoms}
    
    for idx1, n_lp1, sym1 in lp_atoms:
        atom1 = mol.GetAtomWithIdx(idx1)
        for neighbor in atom1.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in lp_idx_map:
                
                idx2, n_lp2, sym2 = lp_idx_map[n_idx]
                # O-O and N-O repulsions are strongest
                if sym1 == 'O' and sym2 == 'O':
                    lp_repulsion += 0.015  
                elif 'O' in [sym1, sym2]:
                    lp_repulsion += 0.008
                else:
                    lp_repulsion += 0.005
    
    features['lp_adjacent_repulsion'] = lp_repulsion
    
    # Hyperconjugation potential: lone pairs adjacent to C-H bonds
    hyperconj_count = 0
    for idx, n_lp, sym in lp_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                # Count C-H bonds on this carbon
                for n2 in neighbor.GetNeighbors():
                    if n2.GetSymbol() == 'H':
                        hyperconj_count += n_lp  
    
    features['lp_hyperconjugation_potential'] = float(hyperconj_count)
    
    # Anomeric effect: O-C-O or N-C-O patterns
    anomeric_count = 0
    acetal_smarts = Chem.MolFromSmarts('[O,N][CH1,CH2][O,N]')
    if acetal_smarts:
        matches = mol.GetSubstructMatches(acetal_smarts)
        anomeric_count = len(matches)
    features['lp_anomeric_centers'] = float(anomeric_count)
    
    # Total lone pair correction estimate (will be refined by ML)
    features['lp_energy_correction'] = (
        lp_energy + 
        lp_repulsion + 
        0.003 * hyperconj_count +  
        0.005 * anomeric_count     
    )
    
    return features

def compute_correlation_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Features to help capture electron correlation effects that scale 
    non-linearly with molecular size and heteroatom content.
    """
    features = {}
    
    n_electrons = sum(atom_counts.get(e, 0) * ATOMIC_PROPS[e]['Z'] for e in atom_counts)
    n_heavy = sum(atom_counts.get(e, 0) for e in ['C', 'N', 'O', 'F'])
    n_O = atom_counts.get('O', 0)
    n_N = atom_counts.get('N', 0)
    
    # Correlation energy scales roughly as N^1.3 for atoms
    features['corr_n_elec_power'] = n_electrons ** 1.3
    
    # Heteroatom-heteroatom correlation (stronger due to electron localization)
    features['corr_hetero_squared'] = (n_O + n_N) ** 2
    
    # O-O pairs have particularly strong correlation
    features['corr_O_pairs'] = n_O * (n_O - 1) / 2 if n_O > 1 else 0
    
    # Cross-term: how "heterogeneous" is the electron distribution
    if n_heavy > 0:
        hetero_frac = (n_O + n_N) / n_heavy
        features['corr_hetero_fraction'] = hetero_frac
        features['corr_hetero_variance'] = hetero_frac * (1 - hetero_frac) * n_heavy
    else:
        features['corr_hetero_fraction'] = 0.0
        features['corr_hetero_variance'] = 0.0
    
    # Size-dependent correlation 
    features['corr_size_effect'] = n_heavy ** 1.5 * (1 + 0.2 * (n_O + n_N))
    
    return features

def compute_nitrogen_environment_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Detailed classification of nitrogen atoms by chemical environment.
    
    Different N types have vastly different electronic contributions:
    - Pyrrole N: lone pair in π-system, not available for donation
    - Pyridine N: lone pair perpendicular to ring, basic
    - Tertiary amine: sp³, pyramidal, 1 available lone pair
    - Amide N: sp², planar, lone pair conjugated with C=O
    - Aniline N: sp², lone pair conjugated with aromatic ring
    """
    features = {}
    
    # Initialize all N-type counters
    n_types = {
        'N_pyrrole': 0,     
        'N_pyridine': 0,     
        'N_aniline': 0,      
        'N_amide': 0,      
        'N_amine_1': 0,      
        'N_amine_2': 0,      
        'N_amine_3': 0,      
        'N_nitrile': 0,      
        'N_nitro': 0,      
        'N_imine': 0,        
    }
    
    # SMARTS patterns for classification
    patterns = {
        'N_pyrrole': '[nH;r5]',                           
        'N_pyrrole_sub': '[n;r5;H0]',                    
        'N_pyridine': '[n;r6;H0]',                        
        'N_aniline': '[NX3;H2][c]',                       
        'N_aniline_sec': '[NX3;H1]([c])[#6]',            
        'N_aniline_tert': '[NX3;H0]([c])([#6])[#6]',     
        'N_amide': '[NX3][CX3](=[OX1])',                  
        'N_amine_1_aliph': '[NX3;H2;!$(NC=O);!$([N][c])]',  
        'N_amine_2_aliph': '[NX3;H1;!$(NC=O);!$([N][c])]',  
        'N_amine_3_aliph': '[NX3;H0;!$(NC=O);!$([N][c]);!$(n)]',  
        'N_nitrile': '[NX1]#[CX2]',                       
        'N_nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]', 
        'N_imine': '[NX2]=[CX3]',                         
    }
    
    for pattern_name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            count = len(matches)
            
            if pattern_name == 'N_pyrrole_sub':
                n_types['N_pyrrole'] += count
            elif pattern_name in ['N_aniline', 'N_aniline_sec', 'N_aniline_tert']:
                n_types['N_aniline'] += count
            elif pattern_name == 'N_amine_1_aliph':
                n_types['N_amine_1'] += count
            elif pattern_name == 'N_amine_2_aliph':
                n_types['N_amine_2'] += count
            elif pattern_name == 'N_amine_3_aliph':
                n_types['N_amine_3'] += count
            elif pattern_name in n_types:
                n_types[pattern_name] += count
    
    for ntype, count in n_types.items():
        features[f'nenv_{ntype}'] = float(count)
    
    # Compute N-type energy contributions 
    N_ENERGY_CONTRIB = {
        'N_pyrrole': -0.02,      
        'N_pyridine': 0.01,      
        'N_aniline': -0.015,    
        'N_amide': -0.025,      
        'N_amine_1': 0.005,     
        'N_amine_2': 0.003,
        'N_amine_3': 0.002,
        'N_nitrile': -0.01,     
        'N_nitro': -0.03,        
        'N_imine': 0.0,
    }
    
    n_energy_correction = sum(n_types[k] * N_ENERGY_CONTRIB.get(k, 0) for k in n_types)
    features['nenv_energy_correction'] = n_energy_correction
    
    
    available_lp = (
        n_types['N_pyridine'] * 1.0 +      
        n_types['N_amine_1'] * 1.0 +
        n_types['N_amine_2'] * 1.0 +
        n_types['N_amine_3'] * 1.0 +
        n_types['N_aniline'] * 0.5 +      
        n_types['N_amide'] * 0.3 +         
        n_types['N_pyrrole'] * 0.1 +      
        n_types['N_nitrile'] * 0.0         
    )
    features['nenv_available_lp'] = available_lp
    
    # N-N interactions 
    total_N = sum(n_types.values())
    features['nenv_total_N'] = float(total_N)
    
    # Aromatic vs aliphatic N ratio
    aromatic_N = n_types['N_pyrrole'] + n_types['N_pyridine']
    aliphatic_N = n_types['N_amine_1'] + n_types['N_amine_2'] + n_types['N_amine_3']
    features['nenv_aromatic_N'] = float(aromatic_N)
    features['nenv_aliphatic_N'] = float(aliphatic_N)
    
    if total_N > 0:
        features['nenv_aromatic_frac'] = aromatic_N / total_N
    else:
        features['nenv_aromatic_frac'] = 0.0
    
    return features

def compute_oxygen_environment_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Detailed classification of oxygen atoms by chemical environment.
    
    Key distinction: Oxygen attached to aromatic rings (anisole-type)
    has fundamentally different electronics than aliphatic ethers.
    """
    features = {}
    
    o_types = {
        'O_carbonyl': 0,        
        'O_carboxyl': 0,       
        'O_ester_carbonyl': 0,  
        'O_ester_ether': 0,     
        'O_amide': 0,           
        'O_alcohol': 0,         
        'O_phenol': 0,         
        'O_ether_aliph': 0,     
        'O_ether_arom': 0,      
        'O_furan': 0,           
        'O_nitro': 0,           
        'O_epoxide': 0,        
    }
    
    patterns = {
        'O_carbonyl_ket': '[CX3;!$(C[OH]);!$(C[O][#6])](=[OX1])[#6]',  
        'O_carboxyl': '[OX2H1][CX3]=[OX1]',                            
        'O_carboxyl_carbonyl': '[OX1]=[CX3][OX2H1]',                   
        'O_ester_carbonyl': '[OX1]=[CX3][OX2][#6]',                    
        'O_ester_ether': '[OX2]([CX3]=[OX1])[#6]',                     
        'O_amide': '[OX1]=[CX3][NX3]',                                 
        'O_alcohol': '[OX2H1][CX4]',                                   
        'O_phenol': '[OX2H1][c]',                                      
        'O_ether_aliph': '[OX2]([CX4])[CX4]',                         
        'O_methoxy_arom': '[OX2]([c])[CH3]',                          
        'O_ether_arom': '[OX2]([c])[#6;!$([CH3])]',                   
        'O_furan': '[oX2;r5]',                                        
        'O_nitro': '[OX1][$([NX3](=O)=O)]',                           
        'O_epoxide': '[OX2;r3]',                                      
    }
    
    for pattern_name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            count = len(matches)
            
            if pattern_name == 'O_carbonyl_ket':
                o_types['O_carbonyl'] += count
            elif pattern_name in ['O_carboxyl', 'O_carboxyl_carbonyl']:
                o_types['O_carboxyl'] += count
            elif pattern_name == 'O_methoxy_arom':
                o_types['O_ether_arom'] += count  
            elif pattern_name in o_types:
                o_types[pattern_name] += count
    
    
    for otype, count in o_types.items():
        features[f'oenv_{otype}'] = float(count)
    
    O_ENERGY_CONTRIB = {
        'O_carbonyl': -0.03,        
        'O_carboxyl': -0.025,
        'O_ester_carbonyl': -0.028,
        'O_ester_ether': -0.01,
        'O_amide': -0.035,          
        'O_alcohol': 0.005,
        'O_phenol': -0.01,          
        'O_ether_aliph': 0.008,     
        'O_ether_arom': -0.02,     
        'O_furan': -0.015,         
        'O_nitro': -0.02,
        'O_epoxide': 0.04,          
    }
    
    o_energy_correction = sum(o_types[k] * O_ENERGY_CONTRIB.get(k, 0) for k in o_types)
    features['oenv_energy_correction'] = o_energy_correction
    
    resonance_O = (
        o_types['O_phenol'] + 
        o_types['O_ether_arom'] +  # This is the methoxy-on-aromatic!
        o_types['O_furan']
    )
    features['oenv_resonance_active'] = float(resonance_O)
    
    available_lp_O = (
        o_types['O_carbonyl'] * 2.0 +
        o_types['O_alcohol'] * 2.0 +
        o_types['O_ether_aliph'] * 2.0 +
        o_types['O_ether_arom'] * 1.2 +   
        o_types['O_phenol'] * 1.5 +        
        o_types['O_furan'] * 0.5 +          
        o_types['O_amide'] * 1.8            
    )
    features['oenv_available_lp'] = available_lp_O
    
    total_O = sum(o_types.values())
    features['oenv_total_O'] = float(total_O)
    
    aromatic_attached_O = o_types['O_ether_arom'] + o_types['O_phenol']
    features['oenv_aromatic_attached'] = float(aromatic_attached_O)
    
    return features

def compute_aromatic_heteroatom_features(mol) -> Dict[str, float]:
    """
    Compute features for heteroatoms directly attached to or in aromatic systems.
    
    This captures the crucial mesomeric (resonance) effects that differentiate
    methoxy-on-benzene from methoxy-on-alkane.
    """
    features = {}   
    
    # Electron donating groups on aromatics
    edg_patterns = {
        'methoxy_arom': '[c][OX2][CH3]',           
        'ethoxy_arom': '[c][OX2][CH2][CH3]',       
        'alkoxy_arom': '[c][OX2][CX4]',           
        'hydroxy_arom': '[c][OX2H1]',              
        'amino_arom': '[c][NX3;H2]',               
        'dialkylamino_arom': '[c][NX3]([#6])[#6]', 
    }
    
    # Electron withdrawing groups on aromatics 
    ewg_patterns = {
        'nitro_arom': '[c][$([NX3](=O)=O)]',       
        'cyano_arom': '[c][CX2]#[NX1]',           
        'carbonyl_arom': '[c][CX3]=O',             
    }
    
    edg_count = 0
    ewg_count = 0
    
    for name, smarts in edg_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            count = len(matches)
            features[f'arom_{name}'] = float(count)
            edg_count += count
    
    for name, smarts in ewg_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            count = len(matches)
            features[f'arom_{name}'] = float(count)
            ewg_count += count
    
    features['arom_edg_total'] = float(edg_count)
    features['arom_ewg_total'] = float(ewg_count)
    features['arom_electronic_balance'] = float(edg_count - ewg_count)
    
    RESONANCE_ENERGY = {
        'methoxy_arom': -0.006,     
        'ethoxy_arom': -0.005,
        'alkoxy_arom': -0.004,
        'hydroxy_arom': -0.008,     
        'amino_arom': -0.010,       
        'dialkylamino_arom': -0.012,
        'nitro_arom': -0.003,       
        'cyano_arom': -0.002,
        'carbonyl_arom': -0.004,
    }
    
    resonance_correction = 0.0
    for name in list(edg_patterns.keys()) + list(ewg_patterns.keys()):
        count = features.get(f'arom_{name}', 0)
        resonance_correction += count * RESONANCE_ENERGY.get(name, 0)
    
    features['arom_resonance_correction'] = resonance_correction
    
    total_substituents = edg_count + ewg_count
    features['arom_total_substituents'] = float(total_substituents)
    
    if total_substituents > 1:
        features['arom_polysubstituted_correction'] = -0.002 * (total_substituents - 1)
    else:
        features['arom_polysubstituted_correction'] = 0.0
    
    return features

def compute_size_scaling_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute features that capture how energy contributions scale with molecular size.
    
    Physics basis:
    - Electron correlation grows non-linearly with system size
    - Heteroatom contributions scale differently in large vs small molecules
    - Bonding network effects become more important in larger systems
    
    These features help the model extrapolate beyond the training set (N_heavy > 9).
    """
    features = {}
    
    n_heavy = sum(atom_counts.get(e, 0) for e in ['C', 'N', 'O', 'F'])
    n_O = atom_counts.get('O', 0)
    n_N = atom_counts.get('N', 0)
    n_C = atom_counts.get('C', 0)
    n_H = atom_counts.get('H', 0)
    n_total = sum(atom_counts.values())
    
    is_extrapolated = n_heavy > 9
    features['size_is_extrapolated'] = float(is_extrapolated)
    
    features['size_beyond_training'] = float(max(0, n_heavy - 9))
    features['size_beyond_training_sq'] = float(max(0, n_heavy - 9) ** 2)
    features['size_O_interaction'] = float(n_O * max(0, n_heavy - 9))
    features['size_O_interaction_sq'] = float(n_O * max(0, n_heavy - 9) ** 2)
    
    if n_heavy > 9:
        o_scaling_factor = 1.0 + 0.03 * (n_heavy - 9)
        features['size_O_scaled'] = float(n_O * o_scaling_factor)
    else:
        features['size_O_scaled'] = float(n_O)
    
    features['size_N_interaction'] = float(n_N * max(0, n_heavy - 9))
    features['size_N_interaction_sq'] = float(n_N * max(0, n_heavy - 9) ** 2)

    n_hetero = n_O + n_N
    features['size_hetero_interaction'] = float(n_hetero * max(0, n_heavy - 9))
    
    # For molecules beyond training, add extra correlation terms
    if n_heavy > 9:
        extra_correlation = (n_heavy ** 1.3) - (9 ** 1.3)
        features['size_extra_correlation'] = extra_correlation       
        hetero_correlation = extra_correlation * (1 + 0.15 * n_O + 0.10 * n_N)
        features['size_hetero_correlation'] = hetero_correlation
    else:
        features['size_extra_correlation'] = 0.0
        features['size_hetero_correlation'] = 0.0
    
    if n_C > 0:
        features['size_hetero_C_ratio'] = float(n_hetero / n_C)        
        if n_heavy > 9:
            framework_stabilization = n_C * 0.001 * max(0, n_heavy - 9)
            features['size_framework_stabilization'] = framework_stabilization
        else:
            features['size_framework_stabilization'] = 0.0
    else:
        features['size_hetero_C_ratio'] = 0.0
        features['size_framework_stabilization'] = 0.0
    
    if n_C > 0:
        h_c_ratio = n_H / n_C
        features['size_h_c_ratio'] = h_c_ratio      
        features['size_unsaturation_proxy'] = max(0, 2.0 - h_c_ratio)
    else:
        features['size_h_c_ratio'] = 0.0
        features['size_unsaturation_proxy'] = 0.0
    
    return features

def compute_zpve_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """Estimate Zero-Point Vibrational Energy (ZPVE) features."""
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
                freq = 1000
        
        freq_sum += freq
    
    n_bends = max(0, n_vib_modes - len(mode_counts))
    avg_bend_freq = 800
    freq_sum += n_bends * avg_bend_freq
    
    zpve_ha = 0.5 * freq_sum * CM_TO_HARTREE
    
    features['zpve_estimated'] = zpve_ha
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


def compute_solvation_features(mol, atom_counts: Dict[str, int]) -> Dict[str, float]:
    """Compute solvation-related features."""
    features = {}
    
    sasa_approx = 0.0
    for elem, count in atom_counts.items():
        r = VDW_RADII.get(elem, 1.5)
        sasa_approx += count * 4 * np.pi * r**2
    
    features['solv_sasa_approx'] = sasa_approx
    
    volume_approx = 0.0
    for elem, count in atom_counts.items():
        r = VDW_RADII.get(elem, 1.5)
        volume_approx += count * (4/3) * np.pi * r**3
    
    features['solv_volume_approx'] = volume_approx
    
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


def compute_scf_bridge_features(mol, atom_counts: Dict[str, int],
                                 zpve_features: Dict[str, float]) -> Dict[str, float]:
    """Features to bridge U0 to SCF energy."""
    features = {}
    
    zpve = zpve_features.get('zpve_estimated', 0.0)
    features['bridge_zpve'] = zpve
    
    n_atoms = sum(atom_counts.values())
    n_dof = 3 * n_atoms - 6
    kT_298 = KB_HARTREE * 298.15
    thermal_corr = 0.5 * n_dof * kT_298
    features['bridge_thermal'] = thermal_corr
    features['bridge_u0_to_scf_corr'] = -zpve
    
    n_electrons = sum(atom_counts.get(e, 0) * ATOMIC_PROPS[e]['Z'] for e in atom_counts)
    features['bridge_n_electrons'] = float(n_electrons)
    features['bridge_corr_proxy'] = n_electrons ** 1.5
    
    return features


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


def get_functional_groups(mol) -> Dict[str, int]:
    """Count functional group occurrences."""
    fg_counts = {}
    for name, smarts in FG_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        fg_counts[name] = len(mol.GetSubstructMatches(pattern)) if pattern else 0
    fg_counts['amine'] = fg_counts['amine_1'] + fg_counts['amine_2'] + fg_counts['amine_3']
    return fg_counts


def get_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Generate Morgan fingerprint for a molecule."""
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


def extract_all_features(smiles: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Extract all physics-enhanced features for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Tuple of (features_dict, metadata_dict) or (None, None) if invalid
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
    
    # Hückel features
    huckel = compute_huckel_features(mol)
    for k, v in huckel.items():
        features[k] = v
    
    # Extended Hückel features
    eht = compute_extended_huckel_features(mol, atom_counts)
    for k, v in eht.items():
        features[k] = v
    
    # ZPVE features
    zpve = compute_zpve_features(mol, atom_counts)
    for k, v in zpve.items():
        features[k] = v
    
    # Solvation features
    solv = compute_solvation_features(mol, atom_counts)
    for k, v in solv.items():
        features[k] = v
    
    # Bridge features
    bridge = compute_scf_bridge_features(mol, atom_counts, zpve)
    for k, v in bridge.items():
        features[k] = v
    
    # Ring features
    ring = compute_ring_features(mol)
    for k, v in ring.items():
        features[k] = v
    
    # Functional groups
    fg_counts = get_functional_groups(mol)
    for fg, cnt in fg_counts.items():
        features[f'fg_{fg}'] = float(cnt)

    # Lone pair features 
    lp_features = compute_lone_pair_features(mol, atom_counts)
    for k, v in lp_features.items():
        features[k] = v
    
    # Nitrogen environment features (detailed N classification)
    nenv_features = compute_nitrogen_environment_features(mol, atom_counts)
    for k, v in nenv_features.items():
        features[k] = v
    
    # Oxygen environment features (detailed O classification)
    oenv_features = compute_oxygen_environment_features(mol, atom_counts)
    for k, v in oenv_features.items():
        features[k] = v
    
    # Aromatic-heteroatom conjugation features
    arom_het_features = compute_aromatic_heteroatom_features(mol)
    for k, v in arom_het_features.items():
        features[k] = v
    # Correlation proxy features 
    corr_features = compute_correlation_features(mol, atom_counts)
    for k, v in corr_features.items():
        features[k] = v
   
    # Size scaling features (for extrapolation)
    size_features = compute_size_scaling_features(mol, atom_counts)
    for k, v in size_features.items():
        features[k] = v
    
    # Additional molecular descriptors
    features['n_bonds'] = float(mol.GetNumBonds())
    features['n_rotatable'] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    features['frac_sp3'] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    
    # Clean up any NaN/Inf values
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
# MODEL LOADING
# =============================================================================

def load_model(model_path: str = 'msep_model.pkl') -> bool:
    """
    Load a pre-trained MSEP model from a pickle file.
    
    Args:
        model_path: Path to the .pkl file containing the trained model
        
    Returns:
        True if successful, raises exception otherwise
        
    Example:
        >>> load_model('msep_model.pkl')
        >>> result = predict_molecule('CCO')
    """
    global MODEL_COMPONENTS, MODEL_PARAMS, _feature_names, _model_loaded
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading MSEP model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    MODEL_COMPONENTS = saved_data['MODEL_COMPONENTS']
    MODEL_PARAMS = saved_data['MODEL_PARAMS']
    _feature_names = saved_data['feature_names']
    
    _model_loaded = True
    
    print(f"    Model version: {MODEL_COMPONENTS.get('version', 'unknown')}")
    print(f"    Features: {len(_feature_names)}")
    print(f"    U0 MAE: {MODEL_PARAMS.get('u0_mae_kcal', 'N/A'):.4f} kcal/mol")
    print(f"    ZPVE MAE: {MODEL_PARAMS.get('zpve_mae_kcal', 'N/A'):.4f} kcal/mol")
    print("    Model loaded successfully ✓")
    
    return True


def is_model_loaded() -> bool:
    """Check if a model has been loaded."""
    return _model_loaded


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_zpve(n_atoms: int, n_H: int, n_bonds: int,
                 features: Dict, X_scaled: np.ndarray) -> float:
    """
    Predict ZPVE using multi-component model.
    
    ZPVE = a×N_atoms + b×N_H + c×N_atoms² + d×N_bonds + intercept + ML_residual
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


def compute_solvation_correction(features: Dict, metadata: Dict,
                                  solvent: str = 'vacuum') -> float:
    """Compute solvation free energy correction (NEGATIVE = stabilization)."""
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


def predict_molecule(
    smiles: str,
    solvent: str = 'vacuum',
    verbose: bool = False,
    return_breakdown: bool = False
) -> Optional[Dict]:
    """
    Predict SCF energy for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        solvent: Solvent name (default: 'vacuum')
        verbose: Print detailed output (default: False)
        return_breakdown: Include energy breakdown in result (default: False)
        
    Returns:
        Dictionary with prediction results, or None if prediction fails
        
    Example:
        >>> result = predict_molecule('CCO', solvent='water')
        >>> print(f"SCF Energy: {result['scf_solvated']:.6f} Ha")
    """
    if not _model_loaded:
        raise RuntimeError("No model loaded. Call load_model() first.")
    
    features, metadata = extract_all_features(smiles)
    if features is None:
        return None
    
    X = np.array([[features.get(fn, 0.0) for fn in _feature_names]])
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
    
    size_corr = np.polyval(MODEL_COMPONENTS['size_coeffs'], min(n_heavy, 9))
    
    fg_corr = 0.0
    for fg, count in metadata['fg_counts'].items():
        if fg in MODEL_COMPONENTS['fg_corrections'] and count > 0:
            fg_corr += MODEL_COMPONENTS['fg_corrections'][fg]

    n_type_corr = 0.0
    if 'n_type_corrections' in MODEL_COMPONENTS:
        n_type_features = ['nenv_N_pyrrole', 'nenv_N_pyridine', 'nenv_N_aniline',
                          'nenv_N_amide', 'nenv_N_amine_1', 'nenv_N_amine_2',
                          'nenv_N_amine_3']
        for ntype_feat in n_type_features:
            count_val = int(features.get(ntype_feat, 0))
            key = f'{ntype_feat}_{count_val}'
            if key in MODEL_COMPONENTS['n_type_corrections']:
                n_type_corr += MODEL_COMPONENTS['n_type_corrections'][key]
    
    o_type_corr = 0.0
    if 'o_type_corrections' in MODEL_COMPONENTS:
        o_type_features = ['oenv_O_carbonyl', 'oenv_O_ether_aliph', 'oenv_O_ether_arom',
                          'oenv_O_alcohol', 'oenv_O_phenol', 'oenv_O_amide']
        for otype_feat in o_type_features:
            count_val = int(features.get(otype_feat, 0))
            key = f'{otype_feat}_{count_val}'
            if key in MODEL_COMPONENTS['o_type_corrections']:
                o_type_corr += MODEL_COMPONENTS['o_type_corrections'][key]
    
    arom_het_corr = 0.0
    if 'arom_het_corrections' in MODEL_COMPONENTS:
        arom_features = ['arom_methoxy_arom', 'arom_alkoxy_arom', 'arom_hydroxy_arom',
                        'arom_amino_arom', 'arom_dialkylamino_arom']
        for arom_feat in arom_features:
            count_val = int(features.get(arom_feat, 0))
            key = f'{arom_feat}_{count_val}'
            if key in MODEL_COMPONENTS['arom_het_corrections']:
                arom_het_corr += MODEL_COMPONENTS['arom_het_corrections'][key]
    
    hetero_corr = 0.0
    if 'heteroatom_corrections' in MODEL_COMPONENTS:
        n_O = metadata['atom_counts'].get('O', 0)
        key = f'O_{n_O}'
        if key in MODEL_COMPONENTS['heteroatom_corrections']:
            hetero_corr += MODEL_COMPONENTS['heteroatom_corrections'][key]
    
    size_hetero_corr = 0.0
    if 'size_hetero_corrections' in MODEL_COMPONENTS:
        n_O = metadata['atom_counts'].get('O', 0)
        # Determine size bin
        if n_heavy < 5:
            size_bin = (1, 5)
        elif n_heavy < 7:
            size_bin = (5, 7)
        elif n_heavy < 9:
            size_bin = (7, 9)
        else:
            size_bin = None  
        
        if size_bin is not None:
            key = f'size_{size_bin[0]}_{size_bin[1]}_O_{n_O}'
            if key in MODEL_COMPONENTS['size_hetero_corrections']:
                size_hetero_corr += MODEL_COMPONENTS['size_hetero_corrections'][key]
    
    extrapolation_corr = 0.0  
    if n_heavy > 9:
        n_beyond = n_heavy - 9
        n_O = metadata['atom_counts'].get('O', 0)
        n_N = metadata['atom_counts'].get('N', 0)
        
        if 'o_size_trend_coeffs' in MODEL_COMPONENTS and n_O > 0:
            coeffs = MODEL_COMPONENTS['o_size_trend_coeffs']
            corr_at_9 = np.polyval(coeffs, 9)
            corr_at_n = np.polyval(coeffs, n_heavy)
            learned_o_corr = (corr_at_n - corr_at_9) * n_O
                       
            if n_heavy > 20:
                damping = 1.0 / (1.0 + 0.03 * (n_heavy - 20))
                learned_o_corr *= damping
            
            extrapolation_corr += learned_o_corr
        else:
            if n_O > 0:
                extrapolation_corr += -0.002 * n_O * n_beyond
                if n_heavy > 15:
                    extrapolation_corr += -0.0005 * n_O * (n_heavy - 15) ** 2
        
        base_size_corr = -0.001 * n_beyond ** 1.5
        extrapolation_corr += base_size_corr
        
        if n_N > 0:
            extrapolation_corr += -0.0008 * n_N * n_beyond
        extrapolation_corr += -0.0002 * (n_O + n_N) * n_beyond ** 1.3
    
    # =================================================================
    # TOTAL FORMATION ENERGY
    # =================================================================
    formation_total = (formation_ml + size_corr + fg_corr + hetero_corr + n_type_corr + o_type_corr + arom_het_corr + size_hetero_corr + extrapolation_corr)
    
    atomic_baseline = metadata['atomic_baseline']
    
    u0 = atomic_baseline + formation_total
    
    zpve = predict_zpve(n_atoms, n_H, n_bonds, features, X_scaled)
    
    e_elec = u0 - zpve
    
    solv_corr = compute_solvation_correction(features, metadata, solvent)
    
    scf_gas = e_elec
    scf_solvated = e_elec + solv_corr
    
    # Uncertainty estimation
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
        coeffs = MODEL_COMPONENTS['zpve_coeffs']
        zpve_baseline = (coeffs['n_atoms'] * n_atoms + coeffs['n_H'] * n_H +
                         coeffs['n_atoms_sq'] * n_atoms**2 + coeffs['n_bonds'] * n_bonds +
                         coeffs['intercept'])
        
        result['breakdown'] = {
            'atomic_baseline': atomic_baseline,
            'formation_ml': formation_ml,
            'size_corr': size_corr,
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
        print(f"{'='*75}")
    
    return result


def predict_batch(smiles_list: List[str], solvent: str = 'vacuum',
                  show_progress: bool = True) -> List[Optional[Dict]]:
    """
    Batch prediction for multiple molecules.
    
    Args:
        smiles_list: List of SMILES strings
        solvent: Solvent name (default: 'vacuum')
        show_progress: Print progress updates (default: True)
        
    Returns:
        List of prediction dictionaries (None for failed predictions)
    """
    if not _model_loaded:
        raise RuntimeError("No model loaded. Call load_model() first.")
    
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
# VERSION INFO
# =============================================================================

__version__ = '1.0.0'
__author__ = 'MSEP Development Team'

if __name__ == '__main__':
    print(f"MSEP Core Library v{__version__}")
    print("Import this module and call load_model() to get started.")
