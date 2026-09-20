#!/usr/bin/env python3
"""Create a CheXpert-compatible external-test CSV from NIH ChestXray14/CXR8.

This maps overlapping labels only. Non-overlapping CheXpert labels are set to 0,
so metrics should be reported only for the five overlap/competition tasks:
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PARENT_LABELS = ['Cardiac_Abnormalities', 'Pulmonary_Opacities', 'Pleural_Findings', 'Focal_Lung_Lesions', 'Fluid_Overload', 'Structural_Other']
CHILD_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Lung Lesion', 'Edema', 'Fracture', 'Support Devices'
]

CXR8_TO_CHEXPERT = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Effusion': 'Pleural Effusion',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
    'Mass': 'Lung Lesion',
    'Nodule': 'Lung Lesion',
    'Fibrosis': 'Lung Opacity',
    'Infiltration': 'Lung Opacity',
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cxr8-root', type=Path, required=True)
    ap.add_argument('--split', choices=['test', 'train_val', 'all'], default='test')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cxr8_root / 'Data_Entry_2017_v2020.csv')
    if args.split != 'all':
        split_file = args.cxr8_root / ('test_list.txt' if args.split == 'test' else 'train_val_list.txt')
        keep = set(x.strip() for x in split_file.read_text().splitlines() if x.strip())
        df = df[df['Image Index'].isin(keep)].copy()

    out = pd.DataFrame()
    out['Path'] = df['Image Index']
    label_sets = df['Finding Labels'].fillna('').map(lambda s: set(s.split('|')) if s else set())
    for col in PARENT_LABELS + CHILD_LABELS:
        out[col] = 0
    for cxr8_label, chex_label in CXR8_TO_CHEXPERT.items():
        out[chex_label] = label_sets.map(lambda labels, lab=cxr8_label: int(lab in labels))

    # Parent hierarchy approximations for compatibility with the hierarchical model's validation code.
    out['Cardiac_Abnormalities'] = out[['Cardiomegaly', 'Enlarged Cardiomediastinum']].max(axis=1)
    out['Pulmonary_Opacities'] = out[['Lung Opacity', 'Consolidation', 'Pneumonia', 'Atelectasis']].max(axis=1)
    out['Pleural_Findings'] = out[['Pleural Effusion', 'Pleural Other', 'Pneumothorax']].max(axis=1)
    out['Focal_Lung_Lesions'] = out[['Lung Lesion']].max(axis=1)
    out['Fluid_Overload'] = out[['Edema', 'Pleural Effusion']].max(axis=1)
    out['Structural_Other'] = out[['Fracture', 'Support Devices']].max(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'WROTE {args.out}')
    print(f'N={len(out)}')
    for lab in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']:
        print(f'{lab}: positives={int(out[lab].sum())} prevalence={out[lab].mean():.6f}')

if __name__ == '__main__':
    main()
