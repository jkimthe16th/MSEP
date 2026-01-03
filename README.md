# MSEP-Physics-Guided-Machine-Learning-for-Molecular-SCF-Energy-Prediction
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
pip install numpy pandas scikit-learn rdkit-pypi requests matplotlib
```

### Dependencies

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- RDKit
- Matplotlib (for report generation)
- requests (for QM9 dataset download)

## Usage

### Training

Run the training script to build the model from the QM9 dataset:

```bash
python MSEP_train.py
```

The training script will:
1. Download the QM9 dataset (or use a local `qm9.csv` file)
2. Extract 316 physics-informed features per molecule
3. Train a 4-stage gradient boosting ensemble
4. Store model components in memory for predictions

**Training output:**
- Model trained on 50,000 molecules
- Automatic feature extraction and scaling
- ZPVE model fitting with multi-component regression

### Prediction

After training, run the prediction script:

```bash
python MSEP_predict.py
```

The prediction script reads `Input_compounds.csv` and outputs:
- `predictions_output.csv`: Detailed predictions
- `validation_report.pdf`: Visual analysis report

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

## Output Format

The output CSV (`predictions_output.csv`) contains:

| Column | Description |
|--------|-------------|
| `Compound` | Compound name |
| `SMILES` | Canonical SMILES |
| `Formula` | Molecular formula |
| `N_Heavy` | Number of heavy atoms |
| `Pred_SCF_Ha` | Predicted SCF energy (Hartree) |
| `Pred_SCF_kcal` | Predicted SCF energy (kcal/mol) |
| `Uncertainty_Ha` | Prediction uncertainty (Hartree) |
| `Pred_ZPVE_Ha` | Predicted ZPVE (Hartree) |
| `Solvent` | Solvent used |
| `Extrapolated` | Whether molecule exceeds training domain |
| `SUCCESS` | Whether prediction is within experimental range |
| `Delta_from_Mean_kcal` | Deviation from experimental mean |

## Programmatic Usage

```python
# After running MSEP_train.py in the same session:

# Single molecule prediction
result = predict_molecule(
    smiles="CN(C)CCc1c[nH]c2ccccc12",  # DMT
    solvent="water",
    verbose=True,
    return_breakdown=True
)

print(f"SCF Energy: {result['scf_solvated']:.6f} Ha")
print(f"ZPVE: {result['zpve']:.6f} Ha")
print(f"Uncertainty: ±{result['uncertainty']:.6f} Ha")

# Batch prediction
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
results = predict_batch(smiles_list, solvent="water")
```

### Return Dictionary Keys

```python
{
    'u0': float,              # Internal energy at 0K (Ha)
    'zpve': float,            # Zero-point vibrational energy (Ha)
    'e_elec': float,          # Electronic energy (Ha)
    'scf_gas': float,         # Gas-phase SCF energy (Ha)
    'solvation': float,       # Solvation correction (Ha)
    'scf_solvated': float,    # Final SCF with solvation (Ha)
    'uncertainty': float,     # Estimated uncertainty (Ha)
    'n_heavy': int,           # Heavy atom count
    'extrapolated': bool,     # True if N_heavy > 9
    'breakdown': dict,        # Detailed energy components (if requested)
}
```

## Supported Elements

The model supports molecules containing: **H, C, N, O, F**

Molecules with other elements will return `None` and be marked as failed.

## Model Architecture

### Feature Engineering (316 features)

1. **Hückel Theory Features**: π-system size, delocalization energy, HOMO-LUMO gap
2. **Extended Hückel Features**: VOIP contributions, Slater exponents
3. **ZPVE Features**: Bond stretching frequencies, vibrational modes
4. **Solvation Features**: SASA, TPSA, H-bond donors/acceptors, polarizability
5. **Ring Features**: Ring count, strain energy, aromaticity
6. **Functional Groups**: Ketones, amides, amines, alcohols, etc.
7. **Morgan Fingerprints**: 256-bit structural fingerprints

### Training Pipeline

```
Stage 1: Huber Regression (robust baseline)
    ↓
Stage 2: Ridge + Polynomial Features (interactions)
    ↓
Stage 3: Gradient Boosting (residual learning)
    ↓
Stage 4: Refinement GB (fine-tuning)
    ↓
Post-hoc Corrections: Size, π-system, functional group
```

## Energy Calculation

The final SCF energy is computed as:

```
SCF = E_baseline + E_formation - ZPVE + ΔG_solvation

where:
  E_baseline = Σ(atomic reference energies)
  E_formation = ML-predicted formation energy
  ZPVE = a×N_atoms + b×N_H + c×N_atoms² + d×N_bonds + residual_ML
  ΔG_solvation = G_cavity + G_electrostatic + G_dispersion + G_hbond
```

## Limitations

- **Training domain**: Best accuracy for molecules with ≤9 heavy atoms
- **Elements**: Only H, C, N, O, F supported
- **Conformers**: Does not predict conformational energy differences
- **Level of theory**: Trained on B3LYP/6-31G(2df,p); other methods may differ

## Citation

If you use this code, please cite:

```
@software{msep2024,
  title={MSEP: Physics-Guided Machine Learning for Molecular SCF Energy Prediction},
  year={2024},
  url={https://github.com/jkimthe16th/MSEP-Physics-Guided-Machine-Learning-for-Molecular-SCF-Energy-Prediction}
}
```

## License

MIT License

## Acknowledgments

- QM9 dataset: Ramakrishnan et al., Scientific Data 1, 140022 (2014)
- RDKit: Open-source cheminformatics toolkit
