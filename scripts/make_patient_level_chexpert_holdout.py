#!/usr/bin/env python3
"""Create a deterministic patient-disjoint CheXpert training hold-out CSV.

This is analysis-only infrastructure: it carves a patient-level split from an
existing hierarchical CheXpert training CSV and writes both the hold-out and the
remaining training CSV plus an audit JSON.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PARENT_LABELS = [
    'Cardiac_Abnormalities', 'Pulmonary_Opacities', 'Pleural_Findings',
    'Focal_Lung_Lesions', 'Fluid_Overload', 'Structural_Other'
]
CHILD_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pleural Effusion', 'Pleural Other',
    'Pneumothorax', 'Lung Lesion', 'Edema', 'Fracture', 'Support Devices'
]


def patient_id(path: str) -> str:
    m = re.search(r'(patient\d+)', str(path))
    if not m:
        raise ValueError(f'Cannot extract patient id from Path={path!r}')
    return m.group(1)


def summarize(df: pd.DataFrame, labels: list[str]) -> dict:
    rows = []
    for lab in labels:
        if lab not in df.columns:
            continue
        y = df[lab].fillna(0).replace(-1, 0).astype(float)
        rows.append({
            'label': lab,
            'n': int(len(df)),
            'positives': int((y == 1).sum()),
            'prevalence': float((y == 1).mean()) if len(df) else 0.0,
        })
    return {'n_images': int(len(df)), 'label_rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-csv', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--target-images', type=int, default=15000)
    ap.add_argument('--seed', type=int, default=20260802)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    if 'Path' not in df.columns:
        raise SystemExit('Input CSV must have Path column')

    df = df.copy()
    df['_patient_id'] = df['Path'].map(patient_id)
    patients = df['_patient_id'].drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(patients)

    selected = []
    count = 0
    patient_counts = df.groupby('_patient_id').size().to_dict()
    for pid in perm:
        selected.append(pid)
        count += int(patient_counts[pid])
        if count >= args.target_images:
            break
    selected = set(selected)

    holdout = df[df['_patient_id'].isin(selected)].copy()
    remainder = df[~df['_patient_id'].isin(selected)].copy()
    overlap = set(holdout['_patient_id']).intersection(set(remainder['_patient_id']))
    assert not overlap, f'Patient leakage: {len(overlap)} overlapping patients'

    holdout_no_aux = holdout.drop(columns=['_patient_id'])
    remainder_no_aux = remainder.drop(columns=['_patient_id'])
    holdout_csv = args.outdir / 'chexpert_train_patient_holdout_powered.csv'
    remainder_csv = args.outdir / 'chexpert_train_remainder_patient_disjoint.csv'
    holdout_no_aux.to_csv(holdout_csv, index=False)
    remainder_no_aux.to_csv(remainder_csv, index=False)

    # Per-label prevalence table for audit and paper methods.
    labels = [x for x in PARENT_LABELS + CHILD_LABELS if x in df.columns]
    prevalence_rows = []
    for split_name, split_df in [('holdout', holdout), ('remainder', remainder), ('full_train_csv', df)]:
        for lab in labels:
            y = split_df[lab].fillna(0).replace(-1, 0).astype(float)
            prevalence_rows.append({
                'split': split_name,
                'level': 'parent' if lab in PARENT_LABELS else 'child',
                'label': lab,
                'n_images': int(len(split_df)),
                'n_patients': int(split_df['_patient_id'].nunique()),
                'positives': int((y == 1).sum()),
                'prevalence': float((y == 1).mean()) if len(split_df) else 0.0,
            })
    prevalence_csv = args.outdir / 'patient_holdout_label_prevalence.csv'
    pd.DataFrame(prevalence_rows).to_csv(prevalence_csv, index=False)

    audit = {
        'input_csv': str(args.input_csv),
        'seed': int(args.seed),
        'target_images': int(args.target_images),
        'holdout_csv': str(holdout_csv),
        'remainder_csv': str(remainder_csv),
        'prevalence_csv': str(prevalence_csv),
        'full_n_images': int(len(df)),
        'full_n_patients': int(df['_patient_id'].nunique()),
        'holdout_n_images': int(len(holdout)),
        'holdout_n_patients': int(holdout['_patient_id'].nunique()),
        'remainder_n_images': int(len(remainder)),
        'remainder_n_patients': int(remainder['_patient_id'].nunique()),
        'patient_overlap_holdout_remainder': int(len(overlap)),
        'labels': labels,
    }
    audit_json = args.outdir / 'patient_holdout_audit.json'
    audit_json.write_text(json.dumps(audit, indent=2))
    print('HOLDOUT_CSV=' + str(holdout_csv), flush=True)
    print('REMAINDER_CSV=' + str(remainder_csv), flush=True)
    print('PREVALENCE_CSV=' + str(prevalence_csv), flush=True)
    print('AUDIT_JSON=' + str(audit_json), flush=True)
    print('AUDIT ' + json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
