# MSEP: Machine Learning SCF Energy Prediction
A physics-informed machine learning pipeline for predicting molecular Self-Consistent Field (SCF) energies at the **B3LYP/6-31G(2df,p)** level of theory. Predictions complete in under 1 second, enabling high-throughput screening of molecular libraries.

![MSEP pipeline](assets/MSEP_pipeline.png)

---

## Table of Contents

- [Key Features](#key-features)
- [Physics Components](#physics-components)
  - [Atomic Baseline Energies](#1-atomic-baseline-energies)
  - [Electronic Structure Features](#2-electronic-structure-features)
  - [Heteroatom Environment Classification](#3-heteroatom-environment-classification)
  - [Electron Correlation Proxies](#4-electron-correlation-proxies)
  - [Multi-Component ZPVE Model](#5-multi-component-zpve-model)
  - [Size-Dependent Corrections](#6-size-dependent-corrections)
  - [Solvation Model](#7-solvation-model-born-like)
  - [Functional Group Corrections](#8-functional-group-corrections)
- [Files](#files)
- [Performance](#performance)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup](#setup)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Python/Jupyter](#pythonjupyter-recommended)
- [Input File Format](#input-file-format)
  - [Example File Format](#example-file-format)
  - [Supported Solvents](#supported-solvents)
- [Supported Elements](#supported-elements)
- [For Developers](#for-developers)
  - [Retraining the Model](#retraining-the-model)
  - [Architecture](#architecture)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

### Key Features

- **Fast predictions**: < 1 second per molecule (~ 57 mols/sec)
- **Physics-informed**: Embeds physical laws directly into the model architecture
- **Generalizable**: Trained on small molecules (≤9 heavy atoms), extends to larger drug-like molecules, tested up to 24 heacy atoms (167% extrapolation)
- **Solvation support**: Predicts energies in various solvents (water, DMSO, methanol, etc.)

### Physics Components

The model incorporates multiple fundamental physical principles organized into hierarchical correction layers:

#### 1. Atomic Baseline Energies
- **B3LYP reference energies** for isolated atoms (H, C, N, O, F)
- Provides the foundation for molecular energy prediction via atom counting

#### 2. Electronic Structure Features

**Hückel Theory (π-systems)**
- π-electron delocalization energy for conjugated and aromatic systems
- HOMO-LUMO gap estimation
- Orbital energy spread for reactivity prediction

**Extended Hückel Theory**
- Valence orbital ionization potentials (VOIP) for all atoms
- Slater orbital exponents for overlap estimation
- Electronegativity and chemical hardness sums

**Lone Pair Interactions**
- Explicit counting of lone pairs by atom type and hybridization (N_sp², N_sp³, O_sp², O_sp³, F)
- Lone pair orbital energy contributions
- Adjacent lone pair repulsion (1,2 and 1,3 interactions)
- Hyperconjugation potential (n→σ* donation to adjacent C-H bonds)
- Anomeric effect detection (O-C-O and N-C-O patterns)

#### 3. Heteroatom Environment Classification

**Nitrogen Environments**
- Pyrrole N (lone pair in π-system)
- Pyridine N (lone pair perpendicular to ring)
- Aniline N (conjugated with aromatic ring)
- Amide N (conjugated with C=O)
- Aliphatic amines (primary, secondary, tertiary)
- Nitrile, nitro, and imine nitrogen

**Oxygen Environments**
- Carbonyl oxygen (ketone, aldehyde)
- Carboxylic acid and ester oxygen
- Amide oxygen
- Alcohols and phenols
- Aliphatic vs aromatic ethers (critical for methoxy groups)
- Furan oxygen (aromatic)
- Epoxide oxygen (ring strain)

**Aromatic-Heteroatom Conjugation**
- Electron-donating groups on aromatics (methoxy, hydroxy, amino)
- Electron-withdrawing groups (nitro, cyano, carbonyl)
- Resonance stabilization energy corrections
- Polysubstituted aromatic interactions

#### 4. Electron Correlation Proxies
- Non-linear scaling with electron count (N^1.3)
- Heteroatom-heteroatom correlation enhancement
- Oxygen pair correlation (particularly strong for multiple O atoms)
- Heteroatom fraction and variance terms

#### 5. Multi-Component ZPVE Model
- Linear baseline from atom count, hydrogen count, and bond count
- Bond-specific vibrational frequencies (C-H, N-H, O-H stretches, etc.)
- Gradient boosting residual model for fine corrections
- Proper handling of high-frequency modes

#### 6. Size-Dependent Corrections

**Within Training Domain (≤9 heavy atoms)**
- Polynomial size correction learned from residuals
- Size-stratified heteroatom corrections (size bins × O count)

**Extrapolation Beyond Training (>9 heavy atoms)**
- Learned O-size trend extrapolation
- Quadratic size correction for large molecules
- Heteroatom-size interaction terms (O×size, N×size)
- Framework stabilization from carbon skeleton
- Damping factors for very large molecules (>20 heavy atoms)

#### 7. Solvation Model (Born-like)
- Kirkwood dielectric continuum for electrostatic solvation
- Cavity formation energy from solvent-accessible surface area
- Dispersion interactions based on molecular polarizability
- Explicit hydrogen bond corrections for protic solvents
- Supported solvents: water, methanol, ethanol, DMSO, acetonitrile, chloroform, hexane, and others

#### 8. Functional Group Corrections
- SMARTS-based detection of 18+ functional groups
- Empirical bias corrections learned from training residuals
- Covers ketones, aldehydes, carboxylic acids, esters, amides, nitriles, amines, alcohols, phenols, ethers, nitro groups, and heterocycles

## Files

| File | Description | Who needs it |
|------|-------------|--------------|
| `msep_core.py` | Core library with all functions | Everyone |
| `msep_model.pkl` | Pre-trained model weights | Everyone |
| `msep_predict.py` | User-facing prediction script | End users |
| `msep_train.py` | Training script (generates model.pkl) | Developers only |

## Performance

| Metric | Result (kcal/mol) |
|--------|-------------------|
| MAE (overall) | 3.42 |
| MAE (N ≤ 7) | 2.10 |
| MAE (N = 9) | 3.57 |
| RMSE | 5.34 |
| Bias | -0.005 |
| P95 Error | 9.17 |
| ZPVE MAE | 2.19 |

*Tested on QM9 dataset (10,000 held-out molecules with ≤9 heavy atoms)*

## Installation

### Requirements

```bash
# Install dependencies
pip install numpy pandas scikit-learn requests

# Install RDKit (required)
conda install -c conda-forge rdkit
# OR
pip install rdkit
```

### Setup

1. Download these files to your working directory:
   - `msep_core.py`
   - `msep_model.pkl`
   - `msep_predict.py`


## Usage

### Command Line

```bash
# Basic usage
python msep_predict.py compounds.csv

# Specify output file
python msep_predict.py compounds.csv --output results.csv

# Specify model path
python msep_predict.py compounds.csv --model /path/to/msep_model.pkl
```

### Python/Jupyter (Recomended)

```python
# Make sure to have `msep_core.py` , `msep_model.pkl` , `msep_predict.py`, and 'Input_compounds.csv' in working directory and all other requirements are installed. 

%run msep_predict.py 

```

## Input File Format

Create a CSV file named `Input_compounds.csv` with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `smiles` | Yes | SMILES string of the molecule |
| `Compound` | No | Compound name/identifier |
| `scf` | No | Experimental SCF energy (Hartree) for validation |
| `solvent` | No | Solvent name (default: vacuum) |

### Example File Format

| Compound | smiles | solvent | scf |
|--------|----------|-------------|-------------|
| `Bufotenin` | `CN(C)CCc1c[nH]c2ccc(O)cc12` | `water` | (In Ha, if available) |

Multiple inputs can exist for the same compound if the user has different SCF energies at different conformers. 
Simply continue to fill out the file, filling all applicable sections (even if repeated). 

### Supported Solvents

- `vacuum` / `gas` (default)
- `water`
- `methanol` / `meoh`
- `ethanol` / `etoh`
- `dmso`
- `acetonitrile` / `mecn`
- `dichloromethane` / `dcm` / `ch2cl2`
- `chloroform` / `chcl3`
- `acetone`
- `thf`
- `hexane`
- `benzene`
- `toluene`


## Supported Elements

The model supports molecules containing: **H, C, N, O, F**

Molecules with other elements will return `None` and be marked as `failed`.

## For Developers

### Retraining the Model

If you need to retrain the model:

```bash
python msep_train.py
```

This will:
1. Download the QM9 dataset
2. Extract features for 50,000 molecules
3. Train the ML models
4. Save to `msep_model.pkl`

Training takes approximately 10-15 minutes.

### Architecture

The model uses a stacked ensemble:
1. **Huber regression** - robust baseline
2. **Ridge regression** with polynomial features
3. **Gradient boosting** (HistGradientBoostingRegressor) - main model
4. **Refinement GB** - captures residuals
5. **Size/FG corrections** - empirical corrections
6. **Heteroatom corrections** - accounts for lone pairs
7. **Extrapolation corrections** - for systems larger than 9 heavy atoms 


## Limitations

- **Training domain**: Best accuracy for molecules with ≤9 heavy atoms
- **Elements**: Only H, C, N, O, F **(current physics features are heavily focused on N, O, weaker in F)**
- **Conformers**: Does not predict conformational energy differences
- **Level of theory**: Trained on B3LYP/6-31G(2df,p); other methods may differ
- **Error by atom type**: Heteroatom-rich molecules tend to have higher prediction errors.

## Citation

If you use this code, please cite:

```
@software{msep2026,
  title={MSEP: Machine Learning SCF Energy Prediction},
  year={2026},
  url={https://github.com/jkimthe16th/MSEP}
}
```

## License

MIT License

## Acknowledgments

- QM9 dataset: Ramakrishnan et al., Scientific Data 1, 140022 (2014)
- RDKit: Open-source cheminformatics toolkit
