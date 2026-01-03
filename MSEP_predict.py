# file: MSEP_predict_fixed_single_value.py
#!/usr/bin/env python
# coding: utf-8
"""
Machine Learning SCF Energy (MSEP) Predictions – fixed experimental SCF parsing

Requires (already loaded in this kernel OR available via your train/predict setup):
  - predict_molecule(smiles, solvent, ...) and MODEL_PARAMS
Optional:
  - predict_batch

Reads "Input_compounds.csv" and writes "predictions_output.csv".

"""

# plotting (used later by PDF section)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

print("=" * 75)
print("Predicting...")
print("=" * 75)
sys.stdout.flush()

# =============================================================================
# PREREQUISITES
# =============================================================================

print("\nChecking prerequisites...")

required = ['predict_molecule', 'MODEL_PARAMS']
for r in required:
    if r not in globals():
        raise RuntimeError(f"Missing: {r}. Ensure training/predict core is loaded (run your train cells or loader).")

try:
    from rdkit import Chem
except ImportError:
    raise RuntimeError("RDKit required (pip install rdkit)")

print("    All found ✓")

HARTREE_TO_KCAL = 627.509474

# =============================================================================
# FILE HANDLING
# =============================================================================

INPUT_CSV = "Input_compounds.csv"
OUTPUT_CSV = "predictions_output.csv"

SEARCH_PATHS_INPUT = [
    INPUT_CSV,
    f"/mnt/user-data/uploads/{INPUT_CSV}",
    f"Project/{INPUT_CSV}",
    f"/mnt/user-data/outputs/{INPUT_CSV}",
]

def find_input_file(name: str = INPUT_CSV) -> Optional[str]:
    """Find input file - check multiple locations."""
    for p in SEARCH_PATHS_INPUT:
        if os.path.exists(p):
            return p
    uploads = "/mnt/user-data/uploads"
    if os.path.exists(uploads):
        for f in os.listdir(uploads):
            if f.lower().endswith('.csv'):
                return os.path.join(uploads, f)
    return None

def canonicalize(smi: str) -> Optional[str]:
    """Convert SMILES to canonical form."""
    if pd.isna(smi) or smi is None:
        return None
    try:
        smi_str = str(smi).strip()
        if not smi_str:
            return None
        mol = Chem.MolFromSmiles(smi_str)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def get_formula(smi: str) -> str:
    """Get molecular formula from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return "?"
        mol_h = Chem.AddHs(mol)
        counts: Dict[str, int] = {}
        for a in mol_h.GetAtoms():
            s = a.GetSymbol()
            counts[s] = counts.get(s, 0) + 1
        parts = []
        for e in ['C', 'H'] + sorted([x for x in counts if x not in ['C', 'H']]):
            if e in counts:
                parts.append(f"{e}{counts[e]}" if counts[e] > 1 else e)
        return "".join(parts)
    except:
        return "?"

def get_atom_counts(smi: str) -> Dict[str, int]:
    """Get atom counts including explicit H."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return {}
        mol_h = Chem.AddHs(mol)
        counts: Dict[str, int] = {}
        for a in mol_h.GetAtoms():
            s = a.GetSymbol()
            counts[s] = counts.get(s, 0) + 1
        return counts
    except Exception:
        return {}

def parse_solvent(val) -> str:
    """Parse solvent value with better handling."""
    if pd.isna(val) or val is None:
        return 'vacuum'
    val = str(val).strip().lower()
    if not val:
        return 'vacuum'
    mapping = {
        'none': 'vacuum', 'gas': 'vacuum', 'vac': 'vacuum',
        'h2o': 'water', 'meoh': 'methanol', 'etoh': 'ethanol',
        'ch2cl2': 'dichloromethane', 'dcm': 'dichloromethane',
        'chcl3': 'chloroform', 'mecn': 'acetonitrile',
    }
    return mapping.get(val, val)

# =============================================================================
# SCF VALUE PARSER  (FIX)
# =============================================================================

_num_pat = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

def parse_scf_cell(cell) -> List[float]:
    """
    Robustly parse experimental SCF(s) from a cell.
    Accepts numbers, lists, ranges, and text with units.
    Returns a list of floats (Hartree).
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []

    # Direct numeric
    if isinstance(cell, (int, float)) and np.isfinite(cell):
        return [float(cell)]

    # String parsing
    s = str(cell).strip()
    if not s:
        return []

    # Detect if the string mentions kcal/mol (then convert to Ha)
    s_lower = s.lower()
    unit_is_kcal = ('kcal' in s_lower) or ('kcal/mol' in s_lower)

    nums = _num_pat.findall(s)
    if not nums:
        return []

    vals = []
    for tok in nums:
        try:
            v = float(tok)
            vals.append(v)
        except:
            continue

    if not vals:
        return []

    # Unit handling:
    # - If explicitly in kcal → convert to Ha
    # - Else keep as-is (Ha)
    if unit_is_kcal:
        vals = [v / HARTREE_TO_KCAL for v in vals]

    # Deduplicate while preserving order
    out: List[float] = []
    seen = set()
    for v in vals:
        if np.isfinite(v) and (v,) not in seen:
            out.append(v)
            seen.add((v,))
    return out

# =============================================================================
# RANGE-BASED SUCCESS CHECK
# =============================================================================

def check_success(pred_scf: float, scf_min: float, scf_max: float,
                  uncertainty: float = 0.0) -> Dict:
    """Check if prediction is within experimental range (handles singletons)."""
    if np.isnan(scf_min) or np.isnan(scf_max):
        return {
            'within_range': None,
            'within_range_with_unc': None,
            'delta_to_range': None,
            'delta_to_range_kcal': None,
        }

    within = (scf_min <= pred_scf <= scf_max)

    pred_low = pred_scf - uncertainty
    pred_high = pred_scf + uncertainty
    overlaps = not (pred_high < scf_min or pred_low > scf_max)

    if pred_scf < scf_min:
        delta = pred_scf - scf_min
    elif pred_scf > scf_max:
        delta = pred_scf - scf_max
    else:
        delta = 0.0

    return {
        'within_range': within,
        'within_range_with_unc': overlaps,
        'delta_to_range': delta,
        'delta_to_range_kcal': delta * HARTREE_TO_KCAL,
    }

# =============================================================================
# MAIN PROCESSING  (FIXED TO TREAT SINGLETONS AS EXP DATA)
# =============================================================================

def process_csv(path: Optional[str] = None) -> pd.DataFrame:
    """Process input CSV and generate predictions (single-value SCF supported)."""
    if path is None:
        path = find_input_file(INPUT_CSV)

    if not path or not os.path.exists(path):
        print(f"\n✗ File not found: {INPUT_CSV}")
        return pd.DataFrame()

    print(f"\nLoading: {path}")
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception as e:
        print(f"    ✗ CSV read error: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip().lower().replace('\ufeff', '').replace(' ', '_') for c in df.columns]

    print(f"    Rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    # Identify columns
    smi_col = next((c for c in ['smiles', 'smi', 'smile', 'canonical_smiles'] if c in df.columns), None)
    if not smi_col:
        raise ValueError("No SMILES column found. Required: 'smiles'")

    name_col = next((c for c in ['compound', 'name', 'molecule', 'mol_name', 'compound_name'] if c in df.columns), None)
    scf_col  = next((c for c in ['scf', 'scf_energy', 'energy', 'e_scf', 'scf_ha', 'total_energy'] if c in df.columns), None)
    solv_col = next((c for c in ['solvent', 'solvation', 'solv', 'medium'] if c in df.columns), None)

    print(f"    SMILES column: '{smi_col}'")
    print(f"    Name column:   '{name_col}'")
    print(f"    SCF column:    '{scf_col}'")
    print(f"    Solvent col:   '{solv_col}'")

    # Canonicalize SMILES
    df['_canon_smi'] = df[smi_col].apply(canonicalize)
    n_valid = df['_canon_smi'].notna().sum()
    print(f"\n    Valid SMILES: {n_valid}/{len(df)}")

    df_valid = df[df['_canon_smi'].notna()].copy()
    unique_smiles = df_valid['_canon_smi'].unique()
    print(f"    Unique molecules: {len(unique_smiles)}")

    print(f"\n    Processing molecules...")
    results: List[Dict] = []

    for idx, smi in enumerate(unique_smiles):
        mol_rows = df_valid[df_valid['_canon_smi'] == smi]

        # Name
        name = smi[:25]
        if name_col:
            names = mol_rows[name_col].dropna().astype(str).tolist()
            names = [n for n in names if n and n.lower() != 'nan']
            if names:
                name = Counter(names).most_common(1)[0][0]

        print(f"        [{idx+1}/{len(unique_smiles)}] {name[:40]}")

        # Experimental SCF values (FIX: robust parse, accept singletons)
        scf_values: List[float] = []
        if scf_col:
            for _, row in mol_rows.iterrows():
                vals = parse_scf_cell(row.get(scf_col))
                if vals:
                    scf_values.extend(vals)

        # Compute stats (singleton → min=max=mean=value)
        n_conformers = len(scf_values)
        scf_min = min(scf_values) if n_conformers >= 1 else np.nan
        scf_max = max(scf_values) if n_conformers >= 1 else np.nan
        scf_mean = float(np.mean(scf_values)) if n_conformers >= 1 else np.nan

        # Solvent (majority vote per molecule)
        solvent = 'vacuum'
        if solv_col:
            solvents = mol_rows[solv_col].dropna().astype(str).tolist()
            solvents = [parse_solvent(s) for s in solvents if s and s.lower() != 'nan']
            if solvents:
                solvent = Counter(solvents).most_common(1)[0][0]

        if scf_col:
            if n_conformers == 1:
                print(f"            SCF (single): {scf_mean:.6f} Ha, solvent={solvent}")
            elif n_conformers > 1:
                print(f"            SCF values: {n_conformers}, range [{scf_min:.6f}, {scf_max:.6f}] Ha, solvent={solvent}")
            else:
                print(f"            SCF: (none provided), solvent={solvent}")

        # Prediction
        pred = predict_molecule(smi, solvent=solvent, verbose=False, return_breakdown=True)
        if pred is None:
            atom_counts = get_atom_counts(smi)
            supported = set(['H', 'C', 'N', 'O', 'F'])
            unsupported = [e for e in atom_counts if e not in supported]
            fail_reason = f"Unsupported elements: {unsupported}" if unsupported else "Unknown error"
            results.append({
                'Compound': name, 'SMILES': smi, 'Status': 'FAILED',
                'Fail_Reason': fail_reason, 'Solvent': solvent,
                'Has_Exp': n_conformers >= 1, 'N_Conformers': n_conformers,
                'SCF_Min_Ha': scf_min, 'SCF_Max_Ha': scf_max, 'SCF_Mean_Ha': scf_mean
            })
            print(f"            ✗ FAILED: {fail_reason}")
            continue

        atom_counts = get_atom_counts(smi)
        counts_str = "; ".join([f"{k}:{atom_counts[k]}" for k in sorted(atom_counts.keys())])

        row = {
            'Compound': name,
            'SMILES': smi,
            'Formula': get_formula(smi),
            'N_Heavy': pred['n_heavy'],
            'N_Atoms': pred['n_atoms'],
            'N_H': pred['n_H'],
            'N_O': pred.get('n_O', atom_counts.get('O', 0)),
            'N_N': pred.get('n_N', atom_counts.get('N', 0)),
            'Solvent': solvent,
            'Extrapolated': 'Yes' if pred['extrapolated'] else 'No',
            'Status': 'OK',
            'Has_Exp': n_conformers >= 1,
            'N_Conformers': n_conformers,
            'Atom_Counts': counts_str,
            'Pred_U0_Ha': pred['u0'],
            'Pred_ZPVE_Ha': pred['zpve'],
            'Pred_E_elec_Ha': pred['e_elec'],
            'Pred_Solvation_Ha': pred['solvation'],
            'Pred_SCF_Ha': pred['scf_solvated'],
            'Uncertainty_Ha': pred['uncertainty'],
            'Pred_SCF_kcal': pred['scf_solvated'] * HARTREE_TO_KCAL,
            'ZPVE_kcal': pred['zpve'] * HARTREE_TO_KCAL,
            'Uncertainty_kcal': pred['uncertainty'] * HARTREE_TO_KCAL,
        }

        if 'breakdown' in pred:
            on_corr = (
                float(pred['breakdown'].get('size_corr', 0.0)) +
                float(pred['breakdown'].get('fg_corr', 0.0)) +
                float(pred['breakdown'].get('pi_corr', 0.0))
            )
            row['ON_Correction_Ha'] = on_corr
            row['ON_Correction_kcal'] = on_corr * HARTREE_TO_KCAL

        # Include element counts for common elements
        for el in ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'S', 'P']:
            if el in atom_counts:
                row[f'N_{el}'] = int(atom_counts[el])

        # Attach experimental stats WHEN WE EVEN HAVE A SINGLE VALUE
        if n_conformers >= 1:
            row.update({
                'SCF_Min_Ha': scf_min,
                'SCF_Max_Ha': scf_max,
                'SCF_Mean_Ha': scf_mean,
                'SCF_Range_Ha': scf_max - scf_min,
                'SCF_Range_kcal': (scf_max - scf_min) * HARTREE_TO_KCAL,
            })
            success = check_success(pred['scf_solvated'], scf_min, scf_max, pred['uncertainty'])
            row.update({
                'SUCCESS': 'YES' if success['within_range'] else 'NO',
                'Within_Range': success['within_range'],
                'Within_Range_With_Unc': success['within_range_with_unc'],
                'Delta_to_Range_Ha': success['delta_to_range'],
                'Delta_to_Range_kcal': success['delta_to_range_kcal'],
                'Delta_from_Mean_Ha': pred['scf_solvated'] - scf_mean,
                'Delta_from_Mean_kcal': (pred['scf_solvated'] - scf_mean) * HARTREE_TO_KCAL,
            })

        results.append(row)

    df_out = pd.DataFrame(results)

    # DISPLAY RESULTS
    print("\n" + "=" * 75)
    print("RESULTS")
    print("=" * 75)

    for idx, row in df_out.iterrows():
        status = row.get('Status', 'OK')
        name = row.get('Compound', 'Unknown')

        if status != 'OK':
            print(f"\n[{idx+1}] {name}: FAILED - {row.get('Fail_Reason', 'Unknown')}")
            if row.get('Has_Exp', False):
                print(f"    (Experimental present, N={row.get('N_Conformers', 0)}, solvent={row.get('Solvent', 'vacuum')})")
            continue

        ext = " [EXT]" if row.get('Extrapolated') == 'Yes' else ""
        print(f"\n[{idx+1}] {name}{ext}")
        print(f"    SMILES:  {row['SMILES'][:60]}")
        print(f"    Formula: {row['Formula']}  N_atoms: {row['N_Atoms']}  Solvent: {row['Solvent']}")
        print(f"    SCF(pred): {row['Pred_SCF_Ha']:.8f} Ha  (±{row['Uncertainty_Ha']:.6f} Ha)")
        if row.get('Has_Exp', False):
            if row['N_Conformers'] == 1:
                print(f"    SCF(exp single): {row['SCF_Mean_Ha']:.8f} Ha")
            else:
                print(f"    SCF(exp range):  [{row['SCF_Min_Ha']:.8f}, {row['SCF_Max_Ha']:.8f}] Ha "
                      f"(mean {row['SCF_Mean_Ha']:.8f})")
            success_str = "✓ YES" if row.get('SUCCESS') == 'YES' else "✗ NO"
            print(f"    Within range: {success_str} | Δ(mean): {row.get('Delta_from_Mean_kcal', 0):+.2f} kcal/mol")

    # SUMMARY
    print(f"\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    n_total = len(df_out)
    n_ok = (df_out['Status'] == 'OK').sum()
    n_failed = (df_out['Status'] != 'OK').sum()
    print(f"    Total: {n_total}  OK: {n_ok}  Failed: {n_failed}")

    if 'Has_Exp' in df_out.columns:
        df_with_exp = df_out[(df_out['Status'] == 'OK') & (df_out['Has_Exp'] == True)]
        print(f"    With experimental data: {len(df_with_exp)}")
        if len(df_with_exp) > 0 and 'SUCCESS' in df_with_exp.columns:
            n_success = (df_with_exp['SUCCESS'] == 'YES').sum()
            n_success_unc = df_with_exp['Within_Range_With_Unc'].sum()
            print(f"    SUCCESS (strict):   {n_success}/{len(df_with_exp)} ({100*n_success/max(len(df_with_exp),1):.1f}%)")
            print(f"    SUCCESS (±σ):       {n_success_unc}/{len(df_with_exp)} ({100*n_success_unc/max(len(df_with_exp),1):.1f}%)")
            if 'Delta_from_Mean_kcal' in df_with_exp.columns:
                mae = df_with_exp['Delta_from_Mean_kcal'].abs().mean()
                bias = df_with_exp['Delta_from_Mean_kcal'].mean()
                print(f"    MAE: {mae:.2f} kcal/mol | Bias: {bias:+.2f} kcal/mol")

    # Save output
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*75}")
    print(f"✓ Saved: {OUTPUT_CSV}")
    print("=" * 75)

    return df_out

# =============================================================================
# REPORT GENERATION
# =============================================================================

PREDICTIONS_CSV = "predictions_output.csv"
OUTPUT_PDF = "validation_report.pdf"
HA_TO_KCAL = 627.509474

SEARCH_PATHS_PRED = [
    PREDICTIONS_CSV,
    "predictions_output.csv",
    f"/mnt/user-data/uploads/{PREDICTIONS_CSV}",
    f"/mnt/user-data/outputs/{PREDICTIONS_CSV}",
    "/mnt/user-data/outputs/predictions_output.csv",
]

def find_predictions_file(name: Optional[str] = None) -> Optional[str]:
    """Find predictions CSV in various locations."""
    for p in SEARCH_PATHS_PRED:
        if os.path.exists(p):
            return p
    outputs = "/mnt/user-data/outputs"
    if os.path.exists(outputs):
        for f in os.listdir(outputs):
            if 'prediction' in f.lower() and f.endswith('.csv'):
                return os.path.join(outputs, f)
    return None

def load_predictions(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load predictions and categorize.

    Returns:
        df_valid: Successfully predicted compounds with experimental data
        df_failed: Failed predictions
        df_no_exp: Valid predictions but no experimental data
    """
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Total rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    if 'Status' in df.columns:
        df_ok = df[df['Status'] == 'OK'].copy()
        df_failed = df[df['Status'] != 'OK'].copy()
    else:
        df_ok = df.copy()
        df_failed = pd.DataFrame()

    if 'SCF_Min_Ha' in df_ok.columns:
        df_valid = df_ok[df_ok['SCF_Min_Ha'].notna()].copy()
        df_no_exp = df_ok[df_ok['SCF_Min_Ha'].isna()].copy()
    else:
        df_valid = pd.DataFrame()
        df_no_exp = df_ok.copy()

    print(f"\n    Categorization:")
    print(f"        Valid + Exp data: {len(df_valid)}")
    print(f"        Valid + No exp:   {len(df_no_exp)}")
    print(f"        Failed:           {len(df_failed)}")

    if len(df_failed) > 0:
        print(f"\n    FAILED COMPOUNDS:")
        for _, row in df_failed.iterrows():
            name = str(row.get('Compound', row.get('SMILES', 'Unknown')))[:40]
            reason = row.get('Fail_Reason', 'Unknown')
            print(f"        ✗ {name}: {reason}")

    return df_valid, df_failed, df_no_exp

def generate_report(df_valid: pd.DataFrame, df_failed: pd.DataFrame = None,
                    df_no_exp: pd.DataFrame = None, output_path: str = OUTPUT_PDF):
    """Generate comprehensive PDF report."""
    if df_failed is None:
        df_failed = pd.DataFrame()
    if df_no_exp is None:
        df_no_exp = pd.DataFrame()

    n_valid = len(df_valid)
    n_failed = len(df_failed)
    n_no_exp = len(df_no_exp)
    n_total = n_valid + n_failed + n_no_exp

    has_exp = n_valid > 0

    print(f"\nGenerating report...")
    print(f"    Total compounds: {n_total}")

    with PdfPages(output_path) as pdf:
        # PAGE 1
        fig1 = plt.figure(figsize=(11, 8.5))
        ax = fig1.add_subplot(111)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.5, 0.95, 'ML SCF Energy Prediction Report (MSEP)', fontsize=22,
                fontweight='bold', ha='center', va='top')
        ax.plot([0.1, 0.9], [0.90, 0.90], 'k-', linewidth=2)

        y = 0.85
        ax.text(0.1, y, f"• Total compounds processed: {n_total}", fontsize=11)
        y -= 0.035
        ax.text(0.1, y, f"• Successful predictions: {n_valid + n_no_exp}", fontsize=11)
        y -= 0.035
        ax.text(0.1, y, f"    - With experimental data: {n_valid}", fontsize=11, color='#1565C0')
        y -= 0.03
        ax.text(0.1, y, f"    - Without experimental data: {n_no_exp}", fontsize=10, color='#757575')
        y -= 0.035
        ax.text(0.1, y, f"• Failed predictions: {n_failed}", fontsize=11,
                color='#C62828' if n_failed > 0 else 'black')

        if has_exp:
            n_success = (df_valid['SUCCESS'] == 'YES').sum() if 'SUCCESS' in df_valid.columns else 0
            n_success_unc = df_valid['Within_Range_With_Unc'].sum() if 'Within_Range_With_Unc' in df_valid.columns else 0

            y -= 0.05
            ax.text(0.1, y, "SUCCESS RATE:", fontsize=11, fontweight='bold')
            y -= 0.035
            ax.text(0.1, y, f"    Strict (within range): {n_success}/{n_valid} ({100*n_success/max(n_valid,1):.1f}%)",
                    fontsize=11, color='#2E7D32')
            y -= 0.03
            ax.text(0.1, y, f"    With uncertainty (±σ): {n_success_unc}/{n_valid} ({100*n_success_unc/max(n_valid,1):.1f}%)",
                    fontsize=11, color='#1565C0')

            if 'Delta_from_Mean_kcal' in df_valid.columns:
                mae = df_valid['Delta_from_Mean_kcal'].abs().mean()
                bias = df_valid['Delta_from_Mean_kcal'].mean()
                y -= 0.05
                ax.text(0.1, y, f"• MAE: {mae:.2f} kcal/mol", fontsize=11)
                y -= 0.03
                ax.text(0.1, y, f"• Bias: {bias:+.2f} kcal/mol", fontsize=11)

        if n_failed > 0:
            y -= 0.06
            ax.text(0.1, y, "FAILED COMPOUNDS:", fontsize=11, fontweight='bold', color='#C62828')
            for i, (_, row) in enumerate(df_failed.head(6).iterrows()):
                y -= 0.025
                name = str(row.get('Compound', row.get('SMILES', 'Unknown')))[:35]
                reason = str(row.get('Fail_Reason', 'Unknown'))[:30]
                ax.text(0.12, y, f"✗ {name}", fontsize=9, color='#C62828')
                ax.text(0.55, y, f"({reason})", fontsize=8, color='#757575')
            if n_failed > 6:
                y -= 0.025
                ax.text(0.12, y, f"... and {n_failed - 6} more", fontsize=9, color='#757575')

        if has_exp:
            table_y = 0.35
            ax.text(0.5, table_y + 0.03, 'COMPARISON SUMMARY', fontsize=12, fontweight='bold', ha='center')
            ax.plot([0.08, 0.92], [table_y, table_y], color='#BDBDBD', linewidth=1)

            headers = ['Compound', 'Atoms', 'O', 'N', 'Solvent', 'Δ(kcal)', '']
            x_pos = [0.08, 0.28, 0.36, 0.44, 0.52, 0.70, 0.85]
            for j, (h, xp) in enumerate(zip(headers, x_pos)):
                ax.text(xp, table_y + 0.015, h, fontsize=9, fontweight='bold')

            for i, (_, row) in enumerate(df_valid.head(10).iterrows()):
                y_row = table_y - 0.025 - i * 0.025
                if row.get('SUCCESS') == 'YES':
                    sym, col = '✓', '#2E7D32'
                elif row.get('Within_Range_With_Unc', False):
                    sym, col = '○', '#FF9800'
                else:
                    sym, col = '✗', '#C62828'

                ax.text(x_pos[0], y_row, str(row.get('Compound', ''))[:15], fontsize=8)
                ax.text(x_pos[1], y_row, str(row.get('N_Heavy', '')), fontsize=8)
                ax.text(x_pos[2], y_row, str(row.get('N_O', 0)), fontsize=8)
                ax.text(x_pos[3], y_row, str(row.get('N_N', 0)), fontsize=8)
                ax.text(x_pos[4], y_row, str(row.get('Solvent', 'vac'))[:8], fontsize=8)
                ax.text(x_pos[5], y_row, f"{row.get('Delta_from_Mean_kcal', 0):+.1f}", fontsize=8)
                ax.text(x_pos[6], y_row, sym, fontsize=11, color=col, fontweight='bold')

            if n_valid > 10:
                y_row = table_y - 0.025 - 10 * 0.025
                ax.text(0.5, y_row, f'... and {n_valid - 10} more', fontsize=8,
                        ha='center', color='#757575')

        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # PAGE 2: Predictions vs Experimental
        if has_exp and n_valid > 0:
            n_plot = min(n_valid, 20)
            df_plot = df_valid.head(n_plot)

            fig_width = max(10, min(n_plot * 0.7, 16))
            fig2, ax2 = plt.subplots(figsize=(fig_width, 6))

            x = np.arange(n_plot)

            exp_mean = df_plot['SCF_Mean_Ha'].values
            exp_min_rel = (df_plot['SCF_Min_Ha'].values - exp_mean) * HA_TO_KCAL
            exp_max_rel = (df_plot['SCF_Max_Ha'].values - exp_mean) * HA_TO_KCAL
            pred_rel = (df_plot['Pred_SCF_Ha'].values - exp_mean) * HA_TO_KCAL
            pred_unc = df_plot['Uncertainty_Ha'].values * HA_TO_KCAL

            bar_w = 0.25
            for i in range(n_plot):
                ax2.fill_between([i - bar_w, i + bar_w],
                                 [exp_min_rel[i]] * 2, [exp_max_rel[i]] * 2,
                                 color='#1565C0', alpha=0.2)
                ax2.plot([i - bar_w, i - bar_w], [exp_min_rel[i], exp_max_rel[i]],
                         color='#1565C0', lw=1.5)
                ax2.plot([i + bar_w, i + bar_w], [exp_min_rel[i], exp_max_rel[i]],
                         color='#1565C0', lw=1.5)

            ax2.plot(x, [0] * n_plot, 'o-', color='#1565C0', lw=2, ms=7,
                     mfc='white', mec='#1565C0', mew=2, label='Exp. mean')

            colors = []
            for _, row in df_plot.iterrows():
                if row.get('SUCCESS') == 'YES':
                    colors.append('#4CAF50')
                elif row.get('Within_Range_With_Unc', False):
                    colors.append('#FF9800')
                else:
                    colors.append('#C62828')

            for i in range(n_plot):
                ax2.errorbar(i, pred_rel[i], yerr=pred_unc[i], fmt='s',
                             color=colors[i], ms=9, mec='white', mew=1.5,
                             capsize=4, capthick=2, elinewidth=2, ecolor=colors[i])

            labels = [str(row.get('Compound', ''))[:12] for _, row in df_plot.iterrows()]
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
            ax2.set_ylabel('Energy vs. exp. mean (kcal/mol)', fontsize=11)
            ax2.set_title('Predicted vs. Experimental SCF', fontsize=13, fontweight='bold')
            ax2.axhline(0, color='#BDBDBD', lw=1, ls='--', alpha=0.5)
            ax2.grid(axis='y', alpha=0.3)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50',
                       markersize=10, label='Success'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF9800',
                       markersize=10, label='Marginal (±σ)'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#C62828',
                       markersize=10, label='Failed'),
                Line2D([0], [0], marker='o', color='#1565C0', markersize=8,
                       markerfacecolor='white', markeredgewidth=2, label='Exp. range'),
            ]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)

        # PAGE 3: Diagnostics
        if has_exp and n_valid > 0:
            fig3, axes = plt.subplots(2, 2, figsize=(11, 8))

            ax = axes[0, 0]
            if 'N_O' in df_valid.columns and 'Delta_from_Mean_kcal' in df_valid.columns:
                df_plot = df_valid[df_valid['Delta_from_Mean_kcal'].notna()]
                if len(df_plot) > 0:
                    ax.scatter(df_plot['N_O'], df_plot['Delta_from_Mean_kcal'].abs(),
                               c='#C62828', s=100, alpha=0.7, edgecolor='white', lw=2)

                    if len(df_plot) > 2:
                        z = np.polyfit(df_plot['N_O'], df_plot['Delta_from_Mean_kcal'].abs(), 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(df_plot['N_O'].min(), df_plot['N_O'].max(), 100)
                        ax.plot(x_line, p(x_line), '--', color='#C62828', alpha=0.5)

                        corr = df_plot['N_O'].corr(df_plot['Delta_from_Mean_kcal'].abs())
                        ax.text(0.95, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                                ha='right', va='top', fontsize=10)

                    ax.set_xlabel('Number of Oxygen Atoms', fontsize=10)
                    ax.set_ylabel('|Error| (kcal/mol)', fontsize=10)
                    ax.set_title('(A) Error vs. Oxygen Count', fontweight='bold')
                    ax.grid(alpha=0.3)

            ax = axes[0, 1]
            if 'N_N' in df_valid.columns and 'Delta_from_Mean_kcal' in df_valid.columns:
                df_plot = df_valid[df_valid['Delta_from_Mean_kcal'].notna()]
                if len(df_plot) > 0:
                    ax.scatter(df_plot['N_N'], df_plot['Delta_from_Mean_kcal'].abs(),
                               c='#1565C0', s=100, alpha=0.7, edgecolor='white', lw=2)

                    if len(df_plot) > 2:
                        z = np.polyfit(df_plot['N_N'], df_plot['Delta_from_Mean_kcal'].abs(), 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(df_plot['N_N'].min(), df_plot['N_N'].max(), 100)
                        ax.plot(x_line, p(x_line), '--', color='#1565C0', alpha=0.5)

                        corr = df_plot['N_N'].corr(df_plot['Delta_from_Mean_kcal'].abs())
                        ax.text(0.95, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                                ha='right', va='top', fontsize=10)

                    ax.set_xlabel('Number of Nitrogen Atoms', fontsize=10)
                    ax.set_ylabel('|Error| (kcal/mol)', fontsize=10)
                    ax.set_title('(B) Error vs. Nitrogen Count', fontweight='bold')
                    ax.grid(alpha=0.3)

            ax = axes[1, 0]
            if 'ErrPerType_kcal' in df_valid.columns:
                atom_errors = {'C': [], 'H': [], 'N': [], 'O': []}
                for _, row in df_valid.iterrows():
                    err_str = row.get('ErrPerType_kcal', '')
                    if pd.notna(err_str) and err_str:
                        for part in str(err_str).split('; '):
                            if ':' in part:
                                try:
                                    el, val = part.split(':')
                                    el = el.strip()
                                    if el in atom_errors:
                                        atom_errors[el].append(abs(float(val)))
                                except:
                                    pass

                elements = [e for e in ['C', 'H', 'N', 'O'] if len(atom_errors[e]) > 0]
                if elements:
                    means = [np.mean(atom_errors[e]) for e in elements]
                    stds = [np.std(atom_errors[e]) for e in elements]
                    colors_bar = {'C': '#424242', 'H': '#90A4AE', 'N': '#1565C0', 'O': '#C62828'}

                    ax.bar(elements, means,
                           color=[colors_bar[e] for e in elements],
                           edgecolor='white', lw=2)
                    ax.errorbar(elements, means, yerr=stds, fmt='none',
                                color='black', capsize=5, capthick=2)

                    ax.set_ylabel('Mean |Error| per Atom (kcal/mol)', fontsize=10)
                    ax.set_title('(C) Error by Atom Type', fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)

            ax = axes[1, 1]
            ax.axis('off')
            ax.text(0.5, 0.95, '(D) Analysis Summary', fontsize=11, fontweight='bold',
                    ha='center', transform=ax.transAxes)

            findings = []
            if 'N_O' in df_valid.columns and len(df_valid) > 0:
                df_O = df_valid[df_valid['N_O'] > 0]
                df_noO = df_valid[df_valid['N_O'] == 0]
                if len(df_O) > 0 and len(df_noO) > 0:
                    mae_O = df_O['Delta_from_Mean_kcal'].abs().mean()
                    mae_noO = df_noO['Delta_from_Mean_kcal'].abs().mean()
                    findings.append(f"• MAE with O: {mae_O:.1f} kcal/mol")
                    findings.append(f"• MAE without O: {mae_noO:.1f} kcal/mol")

            if 'N_N' in df_valid.columns and len(df_valid) > 0:
                df_N = df_valid[df_valid['N_N'] > 0]
                df_noN = df_valid[df_valid['N_N'] == 0]
                if len(df_N) > 0 and len(df_noN) > 0:
                    mae_N = df_N['Delta_from_Mean_kcal'].abs().mean()
                    mae_noN = df_noN['Delta_from_Mean_kcal'].abs().mean()
                    findings.append(f"• MAE with N: {mae_N:.1f} kcal/mol")
                    findings.append(f"• MAE without N: {mae_noN:.1f} kcal/mol")

            findings.append("")
            findings.append("Heteroatom-rich molecules tend")
            findings.append("to have higher prediction errors.")
            findings.append("")
            findings.append("Includes O/N environment")
            findings.append("corrections to mitigate this.")

            ax.text(0.1, 0.8, '\n'.join(findings), fontsize=10, va='top',
                    transform=ax.transAxes, linespacing=1.5)

            plt.tight_layout()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)

        # PAGE 4+: Detailed results
        all_dfs = []
        if len(df_valid) > 0:
            df_valid_copy = df_valid.copy()
            df_valid_copy['_category'] = 'valid'
            all_dfs.append(df_valid_copy)
        if df_no_exp is not None and len(df_no_exp) > 0:
            df_no_exp_copy = df_no_exp.copy()
            df_no_exp_copy['_category'] = 'no_exp'
            all_dfs.append(df_no_exp_copy)
        if df_failed is not None and len(df_failed) > 0:
            df_failed_copy = df_failed.copy()
            df_failed_copy['_category'] = 'failed'
            all_dfs.append(df_failed_copy)

        if all_dfs:
            all_compounds = pd.concat(all_dfs, ignore_index=True)
        else:
            all_compounds = pd.DataFrame()

        if len(all_compounds) > 0:
            rows_per_page = 4
            n_pages = (len(all_compounds) + rows_per_page - 1) // rows_per_page

            for page_idx in range(n_pages):
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                ax.text(0.5, 0.97, f'Detailed Results (Page {page_idx + 1}/{n_pages})',
                        fontsize=13, fontweight='bold', ha='center')
                ax.plot([0.05, 0.95], [0.94, 0.94], 'k-', lw=1)

                start = page_idx * rows_per_page
                end = min(start + rows_per_page, len(all_compounds))

                for i, (_, row) in enumerate(all_compounds.iloc[start:end].iterrows()):
                    y = 0.88 - i * 0.22

                    name = str(row.get('Compound', row.get('SMILES', f'Compound_{start + i + 1}')))[:45]
                    category = row.get('_category', 'unknown')
                    status = row.get('Status', 'OK')

                    if status != 'OK' or category == 'failed':
                        sym, col = 'FAILED', '#C62828'
                    elif category == 'no_exp':
                        sym, col = '—', '#757575'
                    elif row.get('SUCCESS', False) == 'YES':
                        sym, col = '✓', '#2E7D32'
                    elif row.get('Within_Range_With_Unc', False):
                        sym, col = '○', '#FF9800'
                    else:
                        sym, col = '✗', '#C62828'

                    ax.text(0.05, y, f'[{start + i + 1}] {name}', fontsize=10, fontweight='bold')
                    ax.text(0.92, y, sym, fontsize=12, fontweight='bold', color=col, ha='right')

                    if status != 'OK' or category == 'failed':
                        reason = row.get('Fail_Reason', 'Unknown error')
                        ax.text(0.07, y - 0.025, f"Status: FAILED", fontsize=9, color='#C62828')
                        ax.text(0.07, y - 0.045, f"Reason: {reason}", fontsize=8, color='#757575')
                        if pd.notna(row.get('SCF_Min_Ha')) if 'SCF_Min_Ha' in row else False:
                            ax.text(0.07, y - 0.065,
                                    f"(Had experimental data: {row.get('N_Conformers', 0)} conformers)",
                                    fontsize=8, color='#757575')
                    else:
                        lines = []
                        formula = row.get('Formula', 'N/A')
                        n_heavy = row.get('N_Heavy', 'N/A')
                        n_O = row.get('N_O', 0)
                        n_N = row.get('N_N', 0)
                        solvent = row.get('Solvent', 'vacuum')

                        lines.append(f"Formula: {formula}  N_heavy: {n_heavy}  N_O: {n_O}  N_N: {n_N}  Solvent: {solvent}")

                        if pd.notna(row.get('Pred_SCF_Ha')) if 'Pred_SCF_Ha' in row else False:
                            pred_scf = row['Pred_SCF_Ha']
                            unc = row.get('Uncertainty_Ha', 0)
                            lines.append(f"Predicted SCF: {pred_scf:.8f} ± {unc:.6f} Ha")

                        if pd.notna(row.get('SCF_Min_Ha')) if 'SCF_Min_Ha' in row else False:
                            scf_min = row['SCF_Min_Ha']
                            scf_max = row['SCF_Max_Ha']
                            n_conf = row.get('N_Conformers', 1)
                            lines.append(f"Exp. SCF: [{scf_min:.6f}, {scf_max:.6f}] Ha ({n_conf} conf.)")

                            delta = row.get('Delta_from_Mean_kcal', 0)
                            lines.append(f"Δ from mean: {delta:+.2f} kcal/mol")
                        else:
                            lines.append("(No experimental data)")

                        for j, line in enumerate(lines):
                            ax.text(0.07, y - 0.025 - j * 0.022, line, fontsize=8, family='monospace')

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"\n✓ Report saved: {output_path}")
    return output_path

def build_report_from_predictions(predictions_csv_path: Optional[str] = None):
    """Helper to load predictions and generate PDF."""
    if predictions_csv_path is None:
        predictions_csv_path = find_predictions_file()
    if not predictions_csv_path or not os.path.exists(predictions_csv_path):
        print("\n⚠ No predictions_output.csv found. Skipping PDF generation.")
        return None
    df_valid, df_failed, df_no_exp = load_predictions(predictions_csv_path)
    if len(df_valid) == 0 and len(df_failed) == 0 and len(df_no_exp) == 0:
        print("\n⚠ No data to report.")
        return None
    return generate_report(df_valid, df_failed, df_no_exp)

# =============================================================================
# AUTO-RUN
# =============================================================================

print("\n" + "-" * 75)
print("Ready!")
print("-" * 75)

input_path = find_input_file(INPUT_CSV)
if input_path:
    print(f"\nFound: {input_path}")
    PREDICTIONS = process_csv(input_path)
else:
    print(f"\n⚠ No input file found. Use process_csv('path/to/file.csv')")
    PREDICTIONS = pd.DataFrame()

print("\n" + "=" * 75)
print("Prediction complete. Generating PDF...")
print("=" * 75)
sys.stdout.flush()

# Build PDF if predictions exist
if os.path.exists(OUTPUT_CSV):
    build_report_from_predictions(OUTPUT_CSV)
else:
    build_report_from_predictions(None)

print("\n" + "=" * 75)
print("PDF generation step finished.")
print("=" * 75)
