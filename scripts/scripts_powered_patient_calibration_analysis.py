#!/usr/bin/env python3
"""Powered CheXpert-only calibration/hierarchy analysis on patient-level holdout predictions.

Inputs are the per-model prediction NPZ produced by run_selective_gpu1_inference.py.
The script splits the holdout into calibration/test partitions by patient ID, fits
per-label temperature/Platt/isotonic calibrators on calibration patients only,
and evaluates on disjoint test patients.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scripts_calibration_paper_analysis as base  # noqa: E402

PARENT_TO_CHILD = {
    'Cardiac_Abnormalities': ['Enlarged Cardiomediastinum', 'Cardiomegaly'],
    'Pulmonary_Opacities': ['Lung Opacity', 'Consolidation', 'Pneumonia', 'Atelectasis'],
    'Pleural_Findings': ['Pleural Effusion', 'Pleural Other', 'Pneumothorax'],
    'Focal_Lung_Lesions': ['Lung Lesion'],
    'Fluid_Overload': ['Edema'],
    'Structural_Other': ['Fracture', 'Support Devices'],
}


def patient_id(path: str) -> str:
    m = re.search(r'(patient\d+)', str(path))
    if not m:
        raise ValueError(f'Cannot extract patient id from {path!r}')
    return m.group(1)


def patient_split(sample_paths: np.ndarray, frac_cal: float, seed: int):
    pids = np.array([patient_id(str(x)) for x in sample_paths])
    unique = np.unique(pids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    # Greedy by patient until approximately frac_cal by image count.
    counts = pd.Series(pids).value_counts().to_dict()
    target = int(round(len(sample_paths) * frac_cal))
    cal_pids, n = [], 0
    for pid in perm:
        cal_pids.append(pid)
        n += int(counts[pid])
        if n >= target:
            break
    cal_pids = set(cal_pids)
    cal_idx = np.where(np.array([p in cal_pids for p in pids]))[0]
    test_idx = np.where(np.array([p not in cal_pids for p in pids]))[0]
    assert not set(pids[cal_idx]).intersection(set(pids[test_idx]))
    return np.sort(cal_idx), np.sort(test_idx), pids


def fit_all(y_cal, p_cal):
    temps, platt, iso = [], [], []
    for j in range(p_cal.shape[1]):
        temps.append(base.fit_temperature(y_cal[:, j], p_cal[:, j]))
        platt.append(base.fit_platt(y_cal[:, j], p_cal[:, j]))
        iso.append(base.fit_isotonic(y_cal[:, j], p_cal[:, j]))
    return temps, platt, iso


def macro_ece_for_bins(labels, y, p, n_bins_list):
    rows = []
    for n_bins in n_bins_list:
        vals = []
        for j, lab in enumerate(labels):
            if len(np.unique(y[:, j])) < 2:
                continue
            vals.append(base.binary_calibration_metrics(y[:, j], p[:, j], n_bins=n_bins, adaptive=False)['ece'])
        rows.append({'n_bins': int(n_bins), 'macro_ece_equal_width': float(np.mean(vals)) if vals else np.nan})
    return rows


def single_vs_ensemble_rows(z, level, labels, test_idx):
    kind = 'child' if level == 'child' else 'parent'
    y = z[f'{kind}_labels'][test_idx].astype(int)
    keys = base.pred_keys(z['_raw_npz'], kind)
    rows = []
    for key in keys:
        p = base.clip_probs(z['_raw_npz'][key][test_idx])
        m = base.evaluate_block('chexpert_powered_patient_test', level, labels, y, {key.replace(f'_{kind}_preds',''): p}, 100, 9100)
        macro = base.macro_summary(m, ['dataset', 'level', 'method']).iloc[0].to_dict()
        rows.append(macro)
    ens_p = z[f'{kind}_probs'][test_idx]
    m = base.evaluate_block('chexpert_powered_patient_test', level, labels, y, {'fused_ensemble': ens_p}, 100, 9200)
    rows.append(base.macro_summary(m, ['dataset', 'level', 'method']).iloc[0].to_dict())
    return rows


def hierarchy_consistency(child_p, parent_p, child_y, parent_y):
    rows = []
    decoded_parent = parent_p.copy()
    for parent_i, (parent, children) in enumerate(PARENT_TO_CHILD.items()):
        child_idx = [base.CHILD_LABELS.index(c) for c in children]
        max_child = child_p[:, child_idx].max(axis=1)
        violation = max_child > parent_p[:, parent_i]
        parent_auc_before = np.nan
        parent_auc_after = np.nan
        if len(np.unique(parent_y[:, parent_i])) >= 2:
            parent_auc_before = roc_auc_score(parent_y[:, parent_i], parent_p[:, parent_i])
            parent_auc_after = roc_auc_score(parent_y[:, parent_i], np.maximum(parent_p[:, parent_i], max_child))
        rows.append({
            'parent': parent,
            'n': int(len(child_p)),
            'violation_count': int(violation.sum()),
            'violation_rate': float(violation.mean()),
            'parent_auc_before': float(parent_auc_before) if not np.isnan(parent_auc_before) else np.nan,
            'parent_auc_after_parent_max_child_decode': float(parent_auc_after) if not np.isnan(parent_auc_after) else np.nan,
        })
        decoded_parent[:, parent_i] = np.maximum(parent_p[:, parent_i], max_child)
    rows.append({
        'parent': 'MACRO',
        'n': int(len(child_p)),
        'violation_count': int(sum(r['violation_count'] for r in rows)),
        'violation_rate': float(np.mean([r['violation_rate'] for r in rows])),
        'parent_auc_before': float(np.nanmean([r['parent_auc_before'] for r in rows])),
        'parent_auc_after_parent_max_child_decode': float(np.nanmean([r['parent_auc_after_parent_max_child_decode'] for r in rows])),
    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chexpert-npz', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260802)
    ap.add_argument('--bootstrap', type=int, default=500)
    ap.add_argument('--cal-frac', type=float, default=0.5)
    ap.add_argument('--ece-bins', default='5,10,15,20')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'figures').mkdir(exist_ok=True)

    raw = np.load(args.chexpert_npz, allow_pickle=True)
    chex = base.load_fused(args.chexpert_npz)
    chex['_raw_npz'] = raw
    sample_paths = raw['sample_paths'] if 'sample_paths' in raw.files else np.arange(len(chex['child_labels'])).astype(str)
    cal_idx, test_idx, pids = patient_split(sample_paths, args.cal_frac, args.seed)

    child_t, child_p, child_i = fit_all(chex['child_labels'][cal_idx], chex['child_probs'][cal_idx])
    parent_t, parent_p, parent_i = fit_all(chex['parent_labels'][cal_idx], chex['parent_probs'][cal_idx])

    child_y = chex['child_labels'][test_idx]
    child_base = chex['child_probs'][test_idx]
    parent_y = chex['parent_labels'][test_idx]
    parent_base = chex['parent_probs'][test_idx]

    child_methods = {
        'baseline_uncalibrated': child_base,
        'chexpert_temperature': base.apply_labelwise(child_t, child_base, 'temperature'),
        'chexpert_platt': base.apply_labelwise(child_p, child_base, 'platt'),
        'chexpert_isotonic': base.apply_labelwise(child_i, child_base, 'isotonic'),
    }
    parent_methods = {
        'baseline_uncalibrated': parent_base,
        'chexpert_temperature': base.apply_labelwise(parent_t, parent_base, 'temperature'),
        'chexpert_platt': base.apply_labelwise(parent_p, parent_base, 'platt'),
        'chexpert_isotonic': base.apply_labelwise(parent_i, parent_base, 'isotonic'),
    }

    metrics = pd.concat([
        base.evaluate_block('chexpert_powered_patient_test', 'child', base.CHILD_LABELS, child_y, child_methods, args.bootstrap, args.seed),
        base.evaluate_block('chexpert_powered_patient_test', 'parent', base.PARENT_LABELS, parent_y, parent_methods, args.bootstrap, args.seed + 1000),
    ], ignore_index=True)
    metrics.to_csv(args.outdir / 'calibration_metrics_long.csv', index=False)
    macro = base.macro_summary(metrics, ['dataset', 'level', 'method'])
    macro.to_csv(args.outdir / 'macro_summary_by_dataset_level_method.csv', index=False)
    macro[macro['level'].isin(['parent', 'child'])].to_csv(args.outdir / 'chexpert_hierarchy_summary.csv', index=False)

    baseline = macro[macro['method'].eq('baseline_uncalibrated')].set_index(['dataset', 'level'])
    deltas = []
    for _, r in macro.iterrows():
        b = baseline.loc[(r['dataset'], r['level'])]
        d = r.to_dict()
        for m in ['ece_equal_width', 'ace_equal_mass', 'brier', 'nll']:
            d[f'delta_{m}_vs_baseline'] = float(r[m] - b[m])
            d[f'pct_delta_{m}_vs_baseline'] = float((r[m] - b[m]) / b[m]) if b[m] else np.nan
        deltas.append(d)
    pd.DataFrame(deltas).to_csv(args.outdir / 'method_deltas_vs_baseline.csv', index=False)

    param_rows = []
    for level, labels, temps, platt, iso in [
        ('child', base.CHILD_LABELS, child_t, child_p, child_i),
        ('parent', base.PARENT_LABELS, parent_t, parent_p, parent_i),
    ]:
        for lab, t, pm, im in zip(labels, temps, platt, iso):
            param_rows.append({'fit_dataset': 'chexpert_powered_patient_cal', 'level': level, 'label': lab, 'temperature': t, 'platt_a': pm.a, 'platt_b': pm.b, 'isotonic_fit': im is not None})
    pd.DataFrame(param_rows).to_csv(args.outdir / 'calibration_parameters.csv', index=False)

    prevalence_rows = []
    for split, idx in [('cal', cal_idx), ('test', test_idx), ('all_holdout', np.arange(len(sample_paths)) )]:
        for level, labels, y in [('child', base.CHILD_LABELS, chex['child_labels'][idx]), ('parent', base.PARENT_LABELS, chex['parent_labels'][idx])]:
            for j, lab in enumerate(labels):
                prevalence_rows.append({'split': split, 'level': level, 'label': lab, 'n': int(len(idx)), 'n_patients': int(len(set(pids[idx]))), 'positives': int(y[:, j].sum()), 'prevalence': float(y[:, j].mean())})
    pd.DataFrame(prevalence_rows).to_csv(args.outdir / 'label_prevalence_by_split.csv', index=False)

    bins = [int(x) for x in args.ece_bins.split(',') if x.strip()]
    bin_rows = []
    for method, probs in child_methods.items():
        for row in macro_ece_for_bins(base.CHILD_LABELS, child_y, probs, bins):
            row.update({'level': 'child', 'method': method})
            bin_rows.append(row)
    for method, probs in parent_methods.items():
        for row in macro_ece_for_bins(base.PARENT_LABELS, parent_y, probs, bins):
            row.update({'level': 'parent', 'method': method})
            bin_rows.append(row)
    pd.DataFrame(bin_rows).to_csv(args.outdir / 'ece_bin_robustness.csv', index=False)

    pd.DataFrame(single_vs_ensemble_rows(chex, 'child', base.CHILD_LABELS, test_idx)).to_csv(args.outdir / 'single_model_vs_ensemble_child.csv', index=False)
    pd.DataFrame(single_vs_ensemble_rows(chex, 'parent', base.PARENT_LABELS, test_idx)).to_csv(args.outdir / 'single_model_vs_ensemble_parent.csv', index=False)
    pd.DataFrame(hierarchy_consistency(child_base, parent_base, child_y, parent_y)).to_csv(args.outdir / 'hierarchy_consistency_violations.csv', index=False)

    base.save_reliability_set(args.outdir / 'figures', 'chexpert_powered_test', base.CHILD_LABELS, child_y, child_base, child_methods['chexpert_temperature'], base.COMPETITION_TASKS)

    meta = {
        'chexpert_npz': str(args.chexpert_npz),
        'n_total_holdout_images': int(len(sample_paths)),
        'n_total_holdout_patients': int(len(set(pids))),
        'n_cal_images': int(len(cal_idx)),
        'n_cal_patients': int(len(set(pids[cal_idx]))),
        'n_test_images': int(len(test_idx)),
        'n_test_patients': int(len(set(pids[test_idx]))),
        'patient_overlap_cal_test': int(len(set(pids[cal_idx]).intersection(set(pids[test_idx])))),
        'split_seed': int(args.seed),
        'bootstrap_replicates': int(args.bootstrap),
        'ece_equal_width_bins_main': 10,
        'ece_bin_robustness_bins': bins,
        'n_models_child': int(chex['n_models_child'][0]),
        'n_models_parent': int(chex['n_models_parent'][0]),
        'notes': 'Temperature scaling has one slope-like degree of freedom on logits; Platt scaling fits both slope and intercept, so it can correct over/under-confidence plus prevalence/intercept shift.',
    }
    (args.outdir / 'metadata.json').write_text(json.dumps(meta, indent=2))

    def row(level, method):
        r = macro[(macro.level == level) & (macro.method == method)].iloc[0]
        return f"ECE={r.ece_equal_width:.6f}, ACE={r.ace_equal_mass:.6f}, Brier={r.brier:.6f}, AUC={r.auc_context:.6f}"
    report = f"""# Powered CheXpert patient-level calibration analysis\n\nStatus: completed on patient-disjoint calibration/test partitions within the new CheXpert training-derived holdout.\n\n## Split\n\n- Holdout total: {meta['n_total_holdout_images']} images / {meta['n_total_holdout_patients']} patients.\n- Calibration partition: {meta['n_cal_images']} images / {meta['n_cal_patients']} patients.\n- Test partition: {meta['n_test_images']} images / {meta['n_test_patients']} patients.\n- Patient overlap between calibration and test: {meta['patient_overlap_cal_test']}.\n\n## Main child-label calibration\n\n- Baseline: {row('child', 'baseline_uncalibrated')}\n- Temperature: {row('child', 'chexpert_temperature')}\n- Platt: {row('child', 'chexpert_platt')}\n- Isotonic: {row('child', 'chexpert_isotonic')}\n\n## Hierarchy summary\n\n- Parent baseline: {row('parent', 'baseline_uncalibrated')}\n- Child baseline: {row('child', 'baseline_uncalibrated')}\n\n## Added reviewer fixes\n\n- Bootstrap CIs are in `calibration_metrics_long.csv`.\n- Main equal-width ECE uses 10 bins; robustness across {bins} bins is in `ece_bin_robustness.csv`.\n- Temperature-vs-Platt explanation: temperature has one slope-like logit scaling parameter, while Platt has slope plus intercept; the intercept lets Platt correct prevalence/base-rate shifts.\n- Single-model-vs-ensemble calibration is in `single_model_vs_ensemble_child.csv` and `single_model_vs_ensemble_parent.csv`.\n- Hierarchy consistency violations are in `hierarchy_consistency_violations.csv`.\n"""
    (args.outdir / 'powered_calibration_analysis_report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
