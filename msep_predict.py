#!/usr/bin/env python
"""
MSEP Prediction Script
======================
Predict SCF energies for molecules using a pre-trained MSEP model.

Requirements:
    - msep_core.py (core library)
    - msep_model.pkl (pre-trained model)
    - Input CSV file with SMILES column

Usage:
    python msep_predict.py [input.csv] [--model model.pkl] [--output output.csv]
    
Example:
    python msep_predict.py compounds.csv --output predictions.csv

Input CSV columns:
    - smiles: SMILES string (required)
    - compound/name: Compound name (optional)
    - scf/scf_energy: Experimental SCF energy in Ha (optional)
    - solvent: Solvent name (optional, default=vacuum)
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from typing import Dict, Optional
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# =============================================================================
# IMPORT MSEP CORE
# =============================================================================

try:
    from msep_core import (
        load_model, predict_molecule, predict_batch,
        is_model_loaded, HARTREE_TO_KCAL, is_valid_smiles
    )
except ImportError:
    print("ERROR: Could not import msep_core.py")
    print("Make sure msep_core.py is in the same directory or in your PYTHONPATH")
    sys.exit(1)

# Import RDKit for SMILES processing
try:
    from rdkit import Chem
except ImportError:
    print("ERROR: RDKit is required. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL_PATH = 'msep_model.pkl'
DEFAULT_INPUT_CSV = 'Input_compounds.csv'
DEFAULT_OUTPUT_CSV = 'predictions_output.csv'
DEFAULT_OUTPUT_PDF = 'validation_report.pdf'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_file(filename: str) -> Optional[str]:
    """Find input file in common locations."""
    search_paths = [
        filename,
        f"./{filename}",
        f"./data/{filename}",
        os.path.expanduser(f"~/{filename}"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            return path

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
        counts = {}
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
    except:
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


def check_success(pred_scf: float, scf_min: float, scf_max: float,
                  uncertainty: float = 0.0) -> Dict:
    """Check if prediction is within experimental range."""
    if np.isnan(scf_min) or np.isnan(scf_max):
        return {
            'within_range': None,
            'within_range_with_unc': None,
            'delta_to_range': None,
            'delta_to_range_kcal': None,
        }

    within = scf_min <= pred_scf <= scf_max

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
# MAIN PROCESSING FUNCTION
# =============================================================================

def generate_pdf_report(df_results: pd.DataFrame, output_path: str = 'validation_report.pdf'):
    """
    Generate comprehensive PDF validation report with visualizations.

    Args:
        df_results: DataFrame with prediction results from process_csv()
        output_path: Path to output PDF file
    """

    print(f"\n    Generating PDF report...")

    # Categorize results
    if 'Status' in df_results.columns:
        df_ok = df_results[df_results['Status'] == 'OK'].copy()
        df_failed = df_results[df_results['Status'] != 'OK'].copy()
    else:
        df_ok = df_results.copy()
        df_failed = pd.DataFrame()

    if 'SCF_Min_Ha' in df_ok.columns:
        df_valid = df_ok[df_ok['SCF_Min_Ha'].notna()].copy()
        df_no_exp = df_ok[df_ok['SCF_Min_Ha'].isna()].copy()
    else:
        df_valid = pd.DataFrame()
        df_no_exp = df_ok.copy()

    n_valid = len(df_valid)
    n_failed = len(df_failed)
    n_no_exp = len(df_no_exp)
    n_total = n_valid + n_failed + n_no_exp
    has_exp = n_valid > 0

    HA_TO_KCAL = 627.509474

    with PdfPages(output_path) as pdf:

        # =================================================================
        # PAGE 1: Summary
        # =================================================================
        fig1 = plt.figure(figsize=(11, 8.5))
        ax = fig1.add_subplot(111)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.5, 0.95, 'MSEP - ML SCF Energy Prediction Report', fontsize=22,
                fontweight='bold', ha='center', va='top')
        ax.plot([0.1, 0.9], [0.90, 0.90], 'k-', linewidth=2)

        y = 0.85
        ax.text(0.1, y, f"• Total compounds processed: {n_total}", fontsize=11)
        y -= 0.035
        ax.text(0.1, y, f"• Successful predictions: {n_valid + n_no_exp}", fontsize=11)
        y -= 0.035
        ax.text(0.1, y, f"    - With experimental data: {n_valid}", fontsize=10, color='#1565C0')
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
            ax.text(0.1, y, f"    Strict (within range): {n_success}/{n_valid} ({100*n_success/n_valid:.1f}%)",
                    fontsize=11, color='#2E7D32')
            y -= 0.03
            ax.text(0.1, y, f"    With uncertainty (±σ): {n_success_unc}/{n_valid} ({100*n_success_unc/n_valid:.1f}%)",
                    fontsize=11, color='#1565C0')

            if 'Delta_from_Mean_kcal' in df_valid.columns:
                mae = df_valid['Delta_from_Mean_kcal'].abs().mean()
                bias = df_valid['Delta_from_Mean_kcal'].mean()
                y -= 0.05
                ax.text(0.1, y, f"• MAE: {mae:.2f} kcal/mol", fontsize=11)
                y -= 0.03
                ax.text(0.1, y, f"• Bias: {bias:+.2f} kcal/mol", fontsize=11)

        # Failed compounds list
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

        # Comparison table
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

        # =================================================================
        # PAGE 2: Predictions vs Experimental Chart
        # =================================================================
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

            colors = []
            for _, row in df_plot.iterrows():
                if row.get('SUCCESS') == 'YES':
                    colors.append('#4CAF50')
                elif row.get('Within_Range_With_Unc', False):
                    colors.append('#FF9800')
                else:
                    colors.append('#C62828')

            bar_w = 0.25

            # Experimental ranges (blue shaded regions)
            for i in range(n_plot):
                ax2.fill_between([i - bar_w, i + bar_w],
                                 [exp_min_rel[i]] * 2, [exp_max_rel[i]] * 2,
                                 color='#1565C0', alpha=0.2)
                ax2.plot([i - bar_w, i - bar_w], [exp_min_rel[i], exp_max_rel[i]],
                         color='#1565C0', lw=1.5)
                ax2.plot([i + bar_w, i + bar_w], [exp_min_rel[i], exp_max_rel[i]],
                         color='#1565C0', lw=1.5)

            # Experimental mean line
            ax2.plot(x, [0] * n_plot, 'o-', color='#1565C0', lw=2, ms=7,
                     mfc='white', mec='#1565C0', mew=2, label='Exp. mean')

            # Predictions with error bars
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

        # =================================================================
        # PAGE 3: Error Analysis (4 subplots)
        # =================================================================
        if has_exp and n_valid > 0:
            fig3, axes = plt.subplots(2, 2, figsize=(11, 8))

            # (A) Error vs O count
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

            # (B) Error vs N count
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

            # (C) Error by atom type bar chart
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

            # (D) Summary text panel
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
            findings.append("MSEP includes O/N environment")
            findings.append("corrections to mitigate this.")

            ax.text(0.1, 0.8, '\n'.join(findings), fontsize=10, va='top',
                    transform=ax.transAxes, linespacing=1.5)

            plt.tight_layout()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)

        # =================================================================
        # PAGE 4+: Detailed Results (4 compounds per page)
        # =================================================================

        # Combine all compounds for detailed pages
        all_dfs = []
        if len(df_valid) > 0:
            df_valid_copy = df_valid.copy()
            df_valid_copy['_category'] = 'valid'
            all_dfs.append(df_valid_copy)
        if len(df_no_exp) > 0:
            df_no_exp_copy = df_no_exp.copy()
            df_no_exp_copy['_category'] = 'no_exp'
            all_dfs.append(df_no_exp_copy)
        if len(df_failed) > 0:
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
                    elif row.get('SUCCESS') == 'YES':
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
                        if pd.notna(row.get('SCF_Min_Ha')):
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

                        lines.append(
                            f"Formula: {formula}  N_heavy: {n_heavy}  N_O: {n_O}  N_N: {n_N}  Solvent: {solvent}"
                        )

                        if pd.notna(row.get('Pred_SCF_Ha')):
                            pred_scf = row['Pred_SCF_Ha']
                            unc = row.get('Uncertainty_Ha', 0)
                            lines.append(f"Predicted SCF: {pred_scf:.8f} ± {unc:.6f} Ha")

                        if pd.notna(row.get('SCF_Min_Ha')):
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

    print(f"    ✓ PDF report saved: {output_path}")
    return output_path


def process_csv(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Process input CSV and generate predictions.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (optional)

    Returns:
        DataFrame with predictions
    """

    print(f"\nLoading: {input_path}")

    try:
        df = pd.read_csv(input_path, sep=None, engine='python')
    except Exception as e:
        print(f"    ✗ CSV read error: {e}")
        return pd.DataFrame()

    # Normalize column names
    df.columns = [str(c).strip().lower().replace('\ufeff', '').replace(' ', '_') for c in df.columns]

    print(f"    Rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    # =========================================================================
    # IDENTIFY COLUMNS
    # =========================================================================

    smi_col = None
    for c in ['smiles', 'smi', 'smile', 'canonical_smiles']:
        if c in df.columns:
            smi_col = c
            break

    if not smi_col:
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(5)
            if any('c' in str(v).lower() or 'C' in str(v) for v in sample):
                smi_col = c
                print(f"    ⚠ Using '{c}' as SMILES column")
                break

    if not smi_col:
        raise ValueError("No SMILES column found")

    print(f"    SMILES column: '{smi_col}'")

    name_col = None
    for c in ['compound', 'name', 'molecule', 'mol_name', 'compound_name']:
        if c in df.columns:
            name_col = c
            break
    print(f"    Name column: '{name_col}'")

    scf_col = None
    for c in ['scf', 'scf_energy', 'energy', 'e_scf', 'scf_ha', 'total_energy']:
        if c in df.columns:
            scf_col = c
            break
    print(f"    SCF column: '{scf_col}'")

    solv_col = None
    for c in ['solvent', 'solvation', 'solv', 'medium']:
        if c in df.columns:
            solv_col = c
            break
    print(f"    Solvent column: '{solv_col}'")

    # =========================================================================
    # CANONICALIZE SMILES
    # =========================================================================

    df['_canon_smi'] = df[smi_col].apply(canonicalize)

    n_valid = df['_canon_smi'].notna().sum()
    n_invalid = df['_canon_smi'].isna().sum()
    print(f"\n    Valid SMILES: {n_valid}/{len(df)}")

    if n_invalid > 0:
        invalid_rows = df[df['_canon_smi'].isna()]
        print(f"    Invalid SMILES ({n_invalid}):")
        for _, row in invalid_rows.head(5).iterrows():
            name = row.get(name_col, 'Unknown') if name_col else 'Unknown'
            smi = row.get(smi_col, 'N/A')
            print(f"        {name}: {str(smi)[:50]}")

    df_valid = df[df['_canon_smi'].notna()].copy()

    # =========================================================================
    # GROUP BY UNIQUE MOLECULE
    # =========================================================================

    unique_smiles = df_valid['_canon_smi'].unique()
    print(f"\n    Unique molecules: {len(unique_smiles)}")

    print(f"\n    Processing molecules...")
    results = []

    for idx, smi in enumerate(unique_smiles):
        mol_rows = df_valid[df_valid['_canon_smi'] == smi]

        # Get compound name
        name = smi[:25]
        if name_col:
            names = mol_rows[name_col].dropna().astype(str).tolist()
            names = [n for n in names if n and n.lower() != 'nan']
            if names:
                name = Counter(names).most_common(1)[0][0]

        print(f"        [{idx+1}] {name[:30]}")

        # Collect experimental SCF values
        scf_values = []
        if scf_col:
            for _, row in mol_rows.iterrows():
                scf_val = row.get(scf_col)
                if pd.notna(scf_val):
                    try:
                        scf_float = float(scf_val)
                        if scf_float < 0:  # SCF energies are negative
                            scf_values.append(scf_float)
                    except (ValueError, TypeError):
                        pass

        n_conformers = len(scf_values)
        scf_min = min(scf_values) if scf_values else np.nan
        scf_max = max(scf_values) if scf_values else np.nan
        scf_mean = np.mean(scf_values) if scf_values else np.nan

        # Get solvent
        solvent = 'vacuum'
        if solv_col:
            solvents = mol_rows[solv_col].dropna().astype(str).tolist()
            solvents = [parse_solvent(s) for s in solvents if s and s.lower() != 'nan']
            if solvents:
                solvent = Counter(solvents).most_common(1)[0][0]

        if scf_col:
            print(f"            SCF data: {n_conformers} conformers, solvent={solvent}")
            if n_conformers > 0:
                print(f"            SCF range: [{scf_min:.6f}, {scf_max:.6f}]")

        # =====================================================================
        # PREDICTION
        # =====================================================================

        pred = predict_molecule(smi, solvent=solvent, verbose=False, return_breakdown=True)

        if pred is None:
            atom_counts = get_atom_counts(smi)
            unsupported = [e for e in atom_counts if e not in ['H', 'C', 'N', 'O', 'F']]

            fail_reason = "Unknown error"
            if unsupported:
                fail_reason = f"Unsupported elements: {unsupported}"

            results.append({
                'Compound': name,
                'SMILES': smi,
                'Status': 'FAILED',
                'Fail_Reason': fail_reason,
                'Solvent': solvent,
                'N_Conformers': n_conformers,
                'SCF_Min_Ha': scf_min,
                'SCF_Max_Ha': scf_max,
            })
            print(f"            ✗ FAILED: {fail_reason}")
            continue

        # Build result row
        atom_counts = get_atom_counts(smi)
        counts_str = "; ".join([f"{k}:{atom_counts[k]}" for k in sorted(atom_counts.keys())])

        row_result = {
            'Compound': name,
            'SMILES': smi,
            'Formula': get_formula(smi),
            'N_Heavy': pred['n_heavy'],
            'N_Atoms': pred['n_atoms'],
            'N_H': pred['n_H'],
            'N_O': atom_counts.get('O', 0),
            'N_N': atom_counts.get('N', 0),
            'Solvent': solvent,
            'Extrapolated': 'Yes' if pred['extrapolated'] else 'No',
            'Status': 'OK',
            'N_Conformers': n_conformers,
            'Atom_Counts': counts_str,

            # Predictions in Hartree
            'Pred_U0_Ha': pred['u0'],
            'Pred_ZPVE_Ha': pred['zpve'],
            'Pred_E_elec_Ha': pred['e_elec'],
            'Pred_Solvation_Ha': pred['solvation'],
            'Pred_SCF_Ha': pred['scf_solvated'],
            'Uncertainty_Ha': pred['uncertainty'],

            # Predictions in kcal/mol
            'Pred_SCF_kcal': pred['scf_solvated'] * HARTREE_TO_KCAL,
            'ZPVE_kcal': pred['zpve'] * HARTREE_TO_KCAL,
            'Uncertainty_kcal': pred['uncertainty'] * HARTREE_TO_KCAL,
        }

        # Add atom counts
        for el in ['H', 'C', 'N', 'O', 'F']:
            if el in atom_counts:
                row_result[f'N_{el}'] = int(atom_counts[el])

        # Compare with experimental data if available
        if not np.isnan(scf_min):
            row_result.update({
                'SCF_Min_Ha': scf_min,
                'SCF_Max_Ha': scf_max,
                'SCF_Mean_Ha': scf_mean,
                'SCF_Range_Ha': scf_max - scf_min,
                'SCF_Range_kcal': (scf_max - scf_min) * HARTREE_TO_KCAL,
            })

            success = check_success(pred['scf_solvated'], scf_min, scf_max, pred['uncertainty'])

            row_result.update({
                'SUCCESS': 'YES' if success['within_range'] else 'NO',
                'Within_Range': success['within_range'],
                'Within_Range_With_Unc': success['within_range_with_unc'],
                'Delta_to_Range_Ha': success['delta_to_range'],
                'Delta_to_Range_kcal': success['delta_to_range_kcal'],
                'Delta_from_Mean_Ha': pred['scf_solvated'] - scf_mean,
                'Delta_from_Mean_kcal': (pred['scf_solvated'] - scf_mean) * HARTREE_TO_KCAL,
            })

            status = "✓" if success['within_range'] else "✗"
            delta = (pred['scf_solvated'] - scf_mean) * HARTREE_TO_KCAL
            print(f"            {status} Δ from mean: {delta:+.2f} kcal/mol")

        results.append(row_result)

    # =========================================================================
    # CREATE OUTPUT DATAFRAME
    # =========================================================================

    df_results = pd.DataFrame(results)

    # Print summary
    n_total = len(df_results)
    n_ok = len(df_results[df_results['Status'] == 'OK'])
    n_failed = len(df_results[df_results['Status'] == 'FAILED'])

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"    Total molecules: {n_total}")
    print(f"    Successful: {n_ok}")
    print(f"    Failed: {n_failed}")

    if 'SUCCESS' in df_results.columns:
        n_with_exp = df_results['SUCCESS'].notna().sum()
        n_success = (df_results['SUCCESS'] == 'YES').sum()
        if n_with_exp > 0:
            print(f"\n    With experimental data: {n_with_exp}")
            print(f"    Within range: {n_success} ({100*n_success/n_with_exp:.1f}%)")

            mae = df_results['Delta_from_Mean_kcal'].abs().mean()
            print(f"    MAE from mean: {mae:.2f} kcal/mol")

    # Save output
    # Save output
    if output_path:
        df_results.to_csv(output_path, index=False)
        print(f"\n    Results saved to: {output_path}")

    # Generate PDF report
    pdf_path = output_path.replace('.csv', '_report.pdf') if output_path else 'validation_report.pdf'
    generate_pdf_report(df_results, pdf_path)

    return df_results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command line usage."""

    parser = argparse.ArgumentParser(
        description='MSEP - Machine Learning SCF Energy Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python msep_predict.py compounds.csv
    python msep_predict.py compounds.csv --output results.csv
    python msep_predict.py compounds.csv --model my_model.pkl
    
Input CSV should have at minimum a 'smiles' column.
Optional columns: compound/name, scf/scf_energy, solvent
        """
    )

    parser.add_argument('input', nargs='?', default=DEFAULT_INPUT_CSV,
                        help=f'Input CSV file (default: {DEFAULT_INPUT_CSV})')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH,
                        help=f'Model file path (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_CSV,
                        help=f'Output CSV file (default: {DEFAULT_OUTPUT_CSV})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed prediction output')

    args = parser.parse_args()

    print("=" * 75)
    print("MSEP - Machine Learning SCF Energy Predictor")
    print("=" * 75)

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    model_path = find_file(args.model)
    if not model_path:
        print(f"\n✗ Model file not found: {args.model}")
        print("  Make sure msep_model.pkl is in the current directory.")
        print("  Or specify path with --model option.")
        sys.exit(1)

    try:
        load_model(model_path)
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        sys.exit(1)

    # =========================================================================
    # FIND INPUT FILE
    # =========================================================================

    input_path = find_file(args.input)
    if not input_path:
        print(f"\n✗ Input file not found: {args.input}")
        print("  Provide a CSV file with a 'smiles' column.")
        sys.exit(1)

    # =========================================================================
    # PROCESS
    # =========================================================================

    try:
        df_results = process_csv(input_path, args.output)
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 75)
    print("PREDICTION COMPLETE")
    print("=" * 75)

    return df_results


# =============================================================================
# INTERACTIVE USAGE
# =============================================================================

def predict_single(smiles: str, solvent: str = 'vacuum', verbose: bool = True):
    """
    Convenience function for interactive use.

    Args:
        smiles: SMILES string
        solvent: Solvent name (default: 'vacuum')
        verbose: Print detailed output (default: True)

    Example:
        >>> predict_single('CCO', solvent='water')
    """
    if not is_model_loaded():
        model_path = find_file(DEFAULT_MODEL_PATH)
        if model_path:
            load_model(model_path)
        else:
            raise RuntimeError(f"Model not found. Call load_model('path/to/model.pkl') first.")

    return predict_molecule(smiles, solvent=solvent, verbose=verbose)


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    main()
