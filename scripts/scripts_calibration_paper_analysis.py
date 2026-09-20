#!/usr/bin/env python3
"""CheXpert/CXR8 multi-label calibration paper analysis.

Implements CHEXPERT_CALIBRATION_PAPER_EXECUTION_PLAN.md on existing NPZ prediction artifacts.
No model training; only deterministic array analysis.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARENT_LABELS = ['Cardiac_Abnormalities', 'Pulmonary_Opacities', 'Pleural_Findings', 'Focal_Lung_Lesions', 'Fluid_Overload', 'Structural_Other']
CHILD_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Lung Lesion', 'Edema', 'Fracture', 'Support Devices'
]
COMPETITION_TASKS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
CXR8_SHARED_CHILD = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']
PRED_ORDER = ['primary', 'vit', 'effnet', 'swinv2', 'third', 'diversity']

EPS = 1e-6


def clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64), EPS, 1 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def pred_keys(z, kind: str) -> List[str]:
    keys = [f'{name}_{kind}_preds' for name in PRED_ORDER if f'{name}_{kind}_preds' in z.files]
    keys += [k for k in z.files if k.endswith(f'_{kind}_preds') and k not in keys]
    return keys


def load_fused(npz: Path) -> Dict[str, np.ndarray]:
    z = np.load(npz, allow_pickle=True)
    pkeys = pred_keys(z, 'parent')
    ckeys = pred_keys(z, 'child')
    if not pkeys or not ckeys:
        raise ValueError(f'Missing prediction keys in {npz}')
    parent = clip_probs(np.stack([z[k] for k in pkeys], axis=0).mean(axis=0))
    child = clip_probs(np.stack([z[k] for k in ckeys], axis=0).mean(axis=0))
    return {
        'parent_probs': parent,
        'child_probs': child,
        'parent_labels': z['parent_labels'].astype(int),
        'child_labels': z['child_labels'].astype(int),
        'parent_member_keys': np.array(pkeys),
        'child_member_keys': np.array(ckeys),
        'n_models_child': np.array([len(ckeys)]),
        'n_models_parent': np.array([len(pkeys)]),
    }


def choose_multilabel_split(y: np.ndarray, frac_cal: float, seed: int, require_cols: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(y)
    n_cal = max(1, min(n - 1, int(round(n * frac_cal))))
    req = list(require_cols)
    for s in range(seed, seed + 5000):
        rng = np.random.default_rng(s)
        perm = rng.permutation(n)
        cal = np.sort(perm[:n_cal])
        test = np.sort(perm[n_cal:])
        ok = True
        for j in req:
            if len(np.unique(y[cal, j])) < 2 or len(np.unique(y[test, j])) < 2:
                ok = False
                break
        if ok:
            return cal, test, s
    return np.sort(np.arange(n)[:n_cal]), np.sort(np.arange(n)[n_cal:]), seed


def binary_calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10, adaptive: bool = False) -> Dict[str, float]:
    y = y.astype(int)
    p = clip_probs(p)
    out: Dict[str, float] = {}
    if len(np.unique(y)) < 2:
        out.update(ece=np.nan, ace=np.nan, mce=np.nan, brier=np.nan, nll=np.nan, auc=np.nan)
        return out
    brier = brier_score_loss(y, p)
    nll = log_loss(y, p, labels=[0, 1])
    auc = roc_auc_score(y, p)
    if adaptive:
        order = np.argsort(p)
        bins = np.array_split(order, n_bins)
    else:
        bins = []
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            if i == n_bins - 1:
                idx = np.where((p >= edges[i]) & (p <= edges[i + 1]))[0]
            else:
                idx = np.where((p >= edges[i]) & (p < edges[i + 1]))[0]
            bins.append(idx)
    ece = 0.0
    mce = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        conf = float(np.mean(p[idx]))
        acc = float(np.mean(y[idx]))
        gap = abs(acc - conf)
        ece += (len(idx) / len(y)) * gap
        mce = max(mce, gap)
    out.update(ece=float(ece), ace=float(ece) if adaptive else np.nan, mce=float(mce), brier=float(brier), nll=float(nll), auc=float(auc))
    return out


def bootstrap_ece_ci(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int, adaptive: bool = False) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(binary_calibration_metrics(y[idx], p[idx], adaptive=adaptive)['ece'])
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return 1.0
    z = logit(p)
    def obj(t: float) -> float:
        return log_loss(y, sigmoid(z / t), labels=[0, 1])
    res = minimize_scalar(obj, bounds=(0.05, 10.0), method='bounded', options={'xatol': 1e-4})
    return float(res.x) if res.success else 1.0


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return sigmoid(logit(p) / max(float(t), 1e-6))


@dataclass
class PlattModel:
    a: float
    b: float
    ok: bool


def fit_platt(y: np.ndarray, p: np.ndarray) -> PlattModel:
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return PlattModel(1.0, 0.0, False)
    x = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(x, y)
    return PlattModel(float(lr.coef_[0, 0]), float(lr.intercept_[0]), True)


def apply_platt(p: np.ndarray, m: PlattModel) -> np.ndarray:
    return sigmoid(m.a * logit(p) + m.b)


def fit_isotonic(y: np.ndarray, p: np.ndarray):
    y = y.astype(int)
    if len(np.unique(y)) < 2 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(p, y)
    return iso


def apply_labelwise(method_models, probs: np.ndarray, method: str) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.float64)
    for j in range(probs.shape[1]):
        m = method_models[j]
        if method == 'temperature':
            out[:, j] = apply_temperature(probs[:, j], m)
        elif method == 'platt':
            out[:, j] = apply_platt(probs[:, j], m)
        elif method == 'isotonic':
            out[:, j] = m.predict(probs[:, j]) if m is not None else probs[:, j]
        else:
            out[:, j] = probs[:, j]
    return clip_probs(out)


def evaluate_block(name: str, level: str, labels: List[str], y: np.ndarray, probs_by_method: Dict[str, np.ndarray], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for method, pmat in probs_by_method.items():
        for j, lab in enumerate(labels):
            ew = binary_calibration_metrics(y[:, j], pmat[:, j], adaptive=False)
            ad = binary_calibration_metrics(y[:, j], pmat[:, j], adaptive=True)
            ci_low, ci_high = bootstrap_ece_ci(y[:, j], pmat[:, j], n_boot=n_boot, seed=seed + j * 17 + len(method), adaptive=False)
            aci_low, aci_high = bootstrap_ece_ci(y[:, j], pmat[:, j], n_boot=n_boot, seed=seed + j * 19 + len(method), adaptive=True)
            rows.append({
                'dataset': name, 'level': level, 'method': method, 'label': lab,
                'n': int(len(y)), 'positives': int(y[:, j].sum()), 'prevalence': float(y[:, j].mean()),
                'ece_equal_width': ew['ece'], 'ece_ci_low': ci_low, 'ece_ci_high': ci_high,
                'ace_equal_mass': ad['ece'], 'ace_ci_low': aci_low, 'ace_ci_high': aci_high,
                'mce_equal_width': ew['mce'], 'brier': ew['brier'], 'nll': ew['nll'], 'auc_context': ew['auc'],
            })
    return pd.DataFrame(rows)


def macro_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metrics = ['ece_equal_width', 'ace_equal_mass', 'mce_equal_width', 'brier', 'nll', 'auc_context']
    out = df.groupby(group_cols, dropna=False)[metrics].mean(numeric_only=True).reset_index()
    counts = df.groupby(group_cols, dropna=False).agg(n_labels=('label', 'count'), total_n=('n', 'sum'), total_pos=('positives', 'sum')).reset_index()
    return out.merge(counts, on=group_cols, how='left')


def reliability_plot(y: np.ndarray, p: np.ndarray, title: str, out: Path, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        if i == n_bins - 1:
            idx = np.where((p >= edges[i]) & (p <= edges[i + 1]))[0]
        else:
            idx = np.where((p >= edges[i]) & (p < edges[i + 1]))[0]
        if len(idx):
            xs.append(float(np.mean(p[idx]))); ys.append(float(np.mean(y[idx]))); ns.append(len(idx))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    if xs:
        ax.scatter(xs, ys, s=np.maximum(20, np.array(ns) / max(ns) * 180), alpha=0.8)
        ax.plot(xs, ys, '-', alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed positive fraction')
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def save_reliability_set(outdir: Path, dataset: str, labels: List[str], y: np.ndarray, base: np.ndarray, temp: np.ndarray, label_subset: List[str]):
    for lab in label_subset:
        if lab not in labels:
            continue
        j = labels.index(lab)
        safe = lab.replace(' ', '_').replace('/', '_')
        reliability_plot(y[:, j], base[:, j], f'{dataset} {lab} baseline', outdir / f'{dataset}_{safe}_baseline.png')
        reliability_plot(y[:, j], temp[:, j], f'{dataset} {lab} temperature', outdir / f'{dataset}_{safe}_temperature.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chexpert-npz', type=Path, required=True)
    ap.add_argument('--cxr8-npz', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260801)
    ap.add_argument('--bootstrap', type=int, default=500)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'figures').mkdir(exist_ok=True)

    chex = load_fused(args.chexpert_npz)
    cxr8 = load_fused(args.cxr8_npz)

    comp_idx = [CHILD_LABELS.index(x) for x in COMPETITION_TASKS]
    shared_idx = [CHILD_LABELS.index(x) for x in CXR8_SHARED_CHILD]

    chex_cal_idx, chex_test_idx, chex_split_seed = choose_multilabel_split(chex['child_labels'], 0.5, args.seed, comp_idx)
    cxr8_cal_idx, cxr8_test_idx, cxr8_split_seed = choose_multilabel_split(cxr8['child_labels'][:, shared_idx], 0.10, args.seed + 100, range(len(shared_idx)))

    # Fit CheXpert labelwise calibrators on child and parent labels.
    def fit_all(y_cal, p_cal):
        temps, platt, iso = [], [], []
        for j in range(p_cal.shape[1]):
            temps.append(fit_temperature(y_cal[:, j], p_cal[:, j]))
            platt.append(fit_platt(y_cal[:, j], p_cal[:, j]))
            iso.append(fit_isotonic(y_cal[:, j], p_cal[:, j]))
        return temps, platt, iso

    chex_child_t, chex_child_p, chex_child_i = fit_all(chex['child_labels'][chex_cal_idx], chex['child_probs'][chex_cal_idx])
    chex_parent_t, chex_parent_p, chex_parent_i = fit_all(chex['parent_labels'][chex_cal_idx], chex['parent_probs'][chex_cal_idx])

    # Evaluate CheXpert held-out test.
    chex_child_test_base = chex['child_probs'][chex_test_idx]
    chex_child_test_y = chex['child_labels'][chex_test_idx]
    chex_parent_test_base = chex['parent_probs'][chex_test_idx]
    chex_parent_test_y = chex['parent_labels'][chex_test_idx]

    chex_child_methods = {
        'baseline_uncalibrated': chex_child_test_base,
        'chexpert_temperature': apply_labelwise(chex_child_t, chex_child_test_base, 'temperature'),
        'chexpert_platt': apply_labelwise(chex_child_p, chex_child_test_base, 'platt'),
        'chexpert_isotonic': apply_labelwise(chex_child_i, chex_child_test_base, 'isotonic'),
    }
    chex_parent_methods = {
        'baseline_uncalibrated': chex_parent_test_base,
        'chexpert_temperature': apply_labelwise(chex_parent_t, chex_parent_test_base, 'temperature'),
        'chexpert_platt': apply_labelwise(chex_parent_p, chex_parent_test_base, 'platt'),
        'chexpert_isotonic': apply_labelwise(chex_parent_i, chex_parent_test_base, 'isotonic'),
    }
    frames = [
        evaluate_block('chexpert_internal_test', 'child', CHILD_LABELS, chex_child_test_y, chex_child_methods, args.bootstrap, args.seed),
        evaluate_block('chexpert_internal_test', 'parent', PARENT_LABELS, chex_parent_test_y, chex_parent_methods, args.bootstrap, args.seed + 1000),
    ]

    # CXR8 shared-label shift experiment.
    cx_y_all = cxr8['child_labels'][:, shared_idx]
    cx_p_all = cxr8['child_probs'][:, shared_idx]
    cx_y_cal, cx_p_cal = cx_y_all[cxr8_cal_idx], cx_p_all[cxr8_cal_idx]
    cx_y_test, cx_p_test = cx_y_all[cxr8_test_idx], cx_p_all[cxr8_test_idx]
    # CheXpert-fit temperatures subset corresponding to CXR8 shared labels.
    chex_shared_t = [chex_child_t[j] for j in shared_idx]
    chex_shared_p = [chex_child_p[j] for j in shared_idx]
    cx_t, cx_platt, cx_iso = fit_all(cx_y_cal, cx_p_cal)
    cx_methods = {
        'baseline_uncalibrated': cx_p_test,
        'chexpert_temperature_transfer': apply_labelwise(chex_shared_t, cx_p_test, 'temperature'),
        'chexpert_platt_transfer': apply_labelwise(chex_shared_p, cx_p_test, 'platt'),
        'cxr8_temperature_target10pct': apply_labelwise(cx_t, cx_p_test, 'temperature'),
        'cxr8_platt_target10pct': apply_labelwise(cx_platt, cx_p_test, 'platt'),
        'cxr8_isotonic_target10pct': apply_labelwise(cx_iso, cx_p_test, 'isotonic'),
    }
    frames.append(evaluate_block('cxr8_external_test90pct', 'shared_child', CXR8_SHARED_CHILD, cx_y_test, cx_methods, args.bootstrap, args.seed + 2000))

    metrics_long = pd.concat(frames, ignore_index=True)
    metrics_long.to_csv(args.outdir / 'calibration_metrics_long.csv', index=False)

    macro = macro_summary(metrics_long, ['dataset', 'level', 'method'])
    macro.to_csv(args.outdir / 'macro_summary_by_dataset_level_method.csv', index=False)

    hierarchy = macro[macro['dataset'].eq('chexpert_internal_test')].copy()
    hierarchy.to_csv(args.outdir / 'chexpert_hierarchy_summary.csv', index=False)

    cxr8_shift = macro[macro['dataset'].eq('cxr8_external_test90pct')].copy().sort_values('ece_equal_width')
    cxr8_shift.to_csv(args.outdir / 'cxr8_shift_summary.csv', index=False)

    # Method deltas vs baseline.
    baseline = macro[macro['method'].eq('baseline_uncalibrated')].set_index(['dataset', 'level'])
    deltas = []
    for _, r in macro.iterrows():
        b = baseline.loc[(r['dataset'], r['level'])]
        d = r.to_dict()
        for m in ['ece_equal_width', 'ace_equal_mass', 'brier', 'nll']:
            d[f'delta_{m}_vs_baseline'] = float(r[m] - b[m])
            d[f'pct_delta_{m}_vs_baseline'] = float((r[m] - b[m]) / b[m]) if b[m] and not math.isnan(b[m]) else np.nan
        deltas.append(d)
    pd.DataFrame(deltas).to_csv(args.outdir / 'method_deltas_vs_baseline.csv', index=False)

    # Calibration parameters.
    param_rows = []
    for level, labels, temps, platt, iso in [
        ('child', CHILD_LABELS, chex_child_t, chex_child_p, chex_child_i),
        ('parent', PARENT_LABELS, chex_parent_t, chex_parent_p, chex_parent_i),
    ]:
        for lab, t, pm, im in zip(labels, temps, platt, iso):
            param_rows.append({'fit_dataset': 'chexpert_cal', 'level': level, 'label': lab, 'temperature': t, 'platt_a': pm.a, 'platt_b': pm.b, 'isotonic_fit': im is not None})
    for lab, t, pm, im in zip(CXR8_SHARED_CHILD, cx_t, cx_platt, cx_iso):
        param_rows.append({'fit_dataset': 'cxr8_cal_10pct', 'level': 'shared_child', 'label': lab, 'temperature': t, 'platt_a': pm.a, 'platt_b': pm.b, 'isotonic_fit': im is not None})
    pd.DataFrame(param_rows).to_csv(args.outdir / 'calibration_parameters.csv', index=False)

    prevalence_rows = []
    for dataset, level, labels, y in [
        ('chexpert_cal', 'child', CHILD_LABELS, chex['child_labels'][chex_cal_idx]),
        ('chexpert_test', 'child', CHILD_LABELS, chex_child_test_y),
        ('chexpert_cal', 'parent', PARENT_LABELS, chex['parent_labels'][chex_cal_idx]),
        ('chexpert_test', 'parent', PARENT_LABELS, chex_parent_test_y),
        ('cxr8_cal_10pct', 'shared_child', CXR8_SHARED_CHILD, cx_y_cal),
        ('cxr8_test_90pct', 'shared_child', CXR8_SHARED_CHILD, cx_y_test),
    ]:
        for j, lab in enumerate(labels):
            prevalence_rows.append({'dataset': dataset, 'level': level, 'label': lab, 'n': len(y), 'positives': int(y[:, j].sum()), 'prevalence': float(y[:, j].mean())})
    pd.DataFrame(prevalence_rows).to_csv(args.outdir / 'label_prevalence_by_split.csv', index=False)

    # Figures.
    save_reliability_set(args.outdir / 'figures', 'chexpert_test', CHILD_LABELS, chex_child_test_y, chex_child_test_base, chex_child_methods['chexpert_temperature'], COMPETITION_TASKS)
    save_reliability_set(args.outdir / 'figures', 'cxr8_test', CXR8_SHARED_CHILD, cx_y_test, cx_p_test, cx_methods['cxr8_temperature_target10pct'], COMPETITION_TASKS)

    # Concise markdown report.
    def fmt_macro_row(df, dataset, level, method):
        r = df[(df.dataset == dataset) & (df.level == level) & (df.method == method)].iloc[0]
        return f"ECE {r.ece_equal_width:.6f}, ACE {r.ace_equal_mass:.6f}, Brier {r.brier:.6f}, AUC context {r.auc_context:.6f}"

    chex_base = fmt_macro_row(macro, 'chexpert_internal_test', 'child', 'baseline_uncalibrated')
    chex_temp = fmt_macro_row(macro, 'chexpert_internal_test', 'child', 'chexpert_temperature')
    chex_platt = fmt_macro_row(macro, 'chexpert_internal_test', 'child', 'chexpert_platt')
    parent_base = fmt_macro_row(macro, 'chexpert_internal_test', 'parent', 'baseline_uncalibrated')
    parent_temp = fmt_macro_row(macro, 'chexpert_internal_test', 'parent', 'chexpert_temperature')
    cx_base = fmt_macro_row(macro, 'cxr8_external_test90pct', 'shared_child', 'baseline_uncalibrated')
    cx_chex_t = fmt_macro_row(macro, 'cxr8_external_test90pct', 'shared_child', 'chexpert_temperature_transfer')
    cx_tgt_t = fmt_macro_row(macro, 'cxr8_external_test90pct', 'shared_child', 'cxr8_temperature_target10pct')

    top_cxr8 = cxr8_shift.iloc[0]
    meta = {
        'chexpert_npz': str(args.chexpert_npz),
        'cxr8_npz': str(args.cxr8_npz),
        'n_models_chexpert_child': int(chex['n_models_child'][0]),
        'n_models_cxr8_child': int(cxr8['n_models_child'][0]),
        'chexpert_n_total': int(len(chex['child_labels'])),
        'chexpert_n_cal': int(len(chex_cal_idx)),
        'chexpert_n_test': int(len(chex_test_idx)),
        'chexpert_split_seed': int(chex_split_seed),
        'cxr8_n_total': int(len(cx_y_all)),
        'cxr8_n_cal': int(len(cxr8_cal_idx)),
        'cxr8_n_test': int(len(cxr8_test_idx)),
        'cxr8_split_seed': int(cxr8_split_seed),
        'competition_tasks': COMPETITION_TASKS,
        'cxr8_shared_child_labels': CXR8_SHARED_CHILD,
        'bootstrap_replicates': int(args.bootstrap),
    }
    (args.outdir / 'metadata.json').write_text(json.dumps(meta, indent=2))

    report = f"""# CheXpert ensemble calibration analysis report\n\nPlan executed: `CHEXPERT_CALIBRATION_PAPER_EXECUTION_PLAN.md`\n\nStatus: executable and completed on available prediction arrays. This uses the available 4-model ensemble predictions, not a 7-model ensemble. No official CheXpert 500 public-test claim is made.\n\n## Data\n\n- CheXpert internal validation: {meta['chexpert_n_total']} samples split into cal={meta['chexpert_n_cal']} and test={meta['chexpert_n_test']} using seed {meta['chexpert_split_seed']}.\n- CXR8 external: {meta['cxr8_n_total']} samples split into target-cal={meta['cxr8_n_cal']} and test={meta['cxr8_n_test']} using seed {meta['cxr8_split_seed']}.\n- CXR8 shared labels: {', '.join(CXR8_SHARED_CHILD)}.\n\n## In-domain CheXpert child-label calibration\n\n- Baseline: {chex_base}\n- Per-label temperature scaling fitted on CheXpert-cal: {chex_temp}\n- Per-label Platt scaling fitted on CheXpert-cal: {chex_platt}\n\n## Hierarchy hook\n\n- Parent baseline: {parent_base}\n- Parent temperature-scaled: {parent_temp}\n- Child baseline: {chex_base}\n- Child temperature-scaled: {chex_temp}\n\nInterpretation: compare `chexpert_hierarchy_summary.csv` for the parent/child calibration gap and whether children benefit more from recalibration.\n\n## CXR8 distribution-shift calibration\n\n- CXR8 uncalibrated baseline: {cx_base}\n- CheXpert-fit temperature transferred to CXR8: {cx_chex_t}\n- CXR8 10% target-domain temperature refit: {cx_tgt_t}\n- Best CXR8 macro ECE method: {top_cxr8['method']} with ECE {top_cxr8['ece_equal_width']:.6f}.\n\n## Main output files\n\n- `calibration_metrics_long.csv` — per-label ECE/ACE/MCE/Brier/NLL/AUC with bootstrap CIs.\n- `macro_summary_by_dataset_level_method.csv` — macro summary by dataset, hierarchy level, method.\n- `method_deltas_vs_baseline.csv` — pre/post deltas.\n- `chexpert_hierarchy_summary.csv` — parent-vs-child hook.\n- `cxr8_shift_summary.csv` — distribution-shift experiment.\n- `calibration_parameters.csv` — learned per-label temperatures and Platt parameters.\n- `label_prevalence_by_split.csv` — sample/positive counts.\n- `figures/` — reliability diagrams for the five competition tasks.\n\n## Framing sentence\n\nUnlike the cross-dataset classification study, this analysis evaluates probability trustworthiness rather than discrimination, and unlike the selective-prediction study, it measures absolute calibration quality rather than case-ranking for abstention.\n"""
    (args.outdir / 'calibration_analysis_report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
