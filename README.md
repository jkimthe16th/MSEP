# MSEP: Machine Learning SCF Energy Prediction
A physics-informed machine learning pipeline for predicting molecular Self-Consistent Field (SCF) energies at the B3LYP/6-31G(2df,p) level of theory. Predictions complete in under 1 second, enabling high-throughput screening of molecular libraries.

### Key Features

- **Fast predictions**: < 1 second per molecule
- **Physics-informed**: Embeds physical laws directly into the model architecture
- **Generalizable**: Trained on small molecules (≤9 heavy atoms), extends to larger drug-like molecules
- **Solvation support**: Predicts energies in various solvents (water, DMSO, methanol, etc.)

### Physics Components

The model incorporates four fundamental physical principles:

1. **Atomic baseline energies**: Size scaling using B3LYP reference energies
2. **Hückel theory**: π-electron delocalization for conjugated systems
3. **Multi-component ZPVE model**: Zero-point vibrational energy from bond frequencies
4. **Born-like solvation corrections**: Solvent effects using dielectric continuum model

## Files

| File | Description | Who needs it |
|------|-------------|--------------|
| `msep_core.py` | Core library with all functions | Everyone |
| `msep_model.pkl` | Pre-trained model weights | Everyone |
| `msep_predict.py` | User-facing prediction script | End users |
| `msep_train.py` | Training script (generates model.pkl) | Developers only |

## Performance

| Metric | Result (kcal/mol) |
|--------|------------------|
| MAE (overall) | 2.8 |
| MAE (N ≤ 7) | 2.2 |
| MAE (N = 9) | 4.2 |
| RMSE | 3.5 |
| Bias | 0.3 |
| P95 Error | 7.2 |
| ZPVE MAE | 0.06 |

*Tested on QM9 dataset (133,885 molecules with ≤9 heavy atoms)*

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

### Python/Jupyter

```python
from msep_core import load_model, predict_molecule, predict_batch

# Load the model (do this once)
load_model('msep_model.pkl')

# Predict a single molecule
result = predict_molecule('CCO', solvent='water', verbose=True)
print(f"SCF Energy: {result['scf_solvated']:.6f} Ha")
print(f"Uncertainty: ±{result['uncertainty']:.6f} Ha")

# Predict multiple molecules
smiles_list = ['C', 'CC', 'CCC', 'c1ccccc1']
results = predict_batch(smiles_list, solvent='vacuum')
```

## Input File Format

Create a CSV file named `Input_compounds.csv` with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `smiles` | Yes | SMILES string of the molecule |
| `compound` | No | Compound name/identifier |
| `scf` | No | Experimental SCF energy (Hartree) for validation |
| `solvent` | No | Solvent name (default: vacuum) |

### Example Input

```csv
smiles,compound,scf,solvent
CCO,ethanol,-154.123456,water
c1ccccc1,benzene,-232.456789,vacuum
CN(C)CCc1c[nH]c2ccccc12,DMT,,water
```

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

Molecules with other elements will return `None` and be marked as failed.

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


## Limitations

- **Training domain**: Best accuracy for molecules with ≤9 heavy atoms
- **Elements**: Only H, C, N, O, F supported
- **Conformers**: Does not predict conformational energy differences
- **Level of theory**: Trained on B3LYP/6-31G(2df,p); other methods may differ
- **Error by atom type**: Heteroatom-rich molecules tend to have higher prediction errors.

## Citation

If you use this code, please cite:

```
@software{msep2024,
  title={MSEP: Machine Learning SCF Energy Prediction},
  year={2025},
  url={https://github.com/jkimthe16th/MSEP}
}
```

## License

MIT License

## Acknowledgments

- QM9 dataset: Ramakrishnan et al., Scientific Data 1, 140022 (2014)
- RDKit: Open-source cheminformatics toolkit
