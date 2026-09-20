#!/usr/bin/env python3
"""Tasks 1-3 hard-gate decision analysis for powered CheXpert calibration paper.

Task 1: single-vs-ensemble calibration decision for whether "ensemble" stays in title.
Task 2: stress-test suspiciously-low ECEs across patient splits and ECE bin counts.
Task 3: prevalence-match parent/child gap to decide whether hierarchy gap claim survives.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression

PARENT_LABELS = [
    'Cardiac_Abnormalities', 'Pulmonary_Opacities', 'Pleural_Findings',
    'Focal_Lung_Lesions', 'Fluid_Overload', 'Structural_Other'
]
CHILD_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pleural Effusion', 'Pleural Other',
    'Pneumothorax', 'Lung Lesion', 'Edema', 'Fracture', 'Support Devices'
]
MODEL_NAMES = ['primary', 'vit', 'effnet', 'swinv2']
EPS = 1e-7


def clip(p): return np.clip(np.asarray(p, dtype=np.float64), EPS, 1-EPS)
def logit(p):
    p = clip(p)
    return np.log(p/(1-p))
def sigmoid(x): return 1/(1+np.exp(-x))

def patient_id(path: str) -> str:
    m = re.search(r'(patient\d+)', str(path))
    if not m: raise ValueError(f'No patient id in {path!r}')
    return m.group(1)

def patient_split(paths, frac_cal, seed):
    pids = np.array([patient_id(str(x)) for x in paths])
    unique = np.unique(pids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    counts = pd.Series(pids).value_counts().to_dict()
    target = int(round(len(paths)*frac_cal))
    cal, n = [], 0
    for pid in perm:
        cal.append(pid); n += int(counts[pid])
        if n >= target: break
    cal = set(cal)
    cal_idx = np.where(np.array([p in cal for p in pids]))[0]
    test_idx = np.where(np.array([p not in cal for p in pids]))[0]
    assert not set(pids[cal_idx]).intersection(set(pids[test_idx]))
    return cal_idx, test_idx, pids

@dataclass
class Platt:
    a: float = 1.0
    b: float = 0.0
    def apply(self, p): return sigmoid(self.a * logit(p) + self.b)

def fit_temp(y, p):
    if len(np.unique(y)) < 2: return 1.0
    x = logit(p); y = y.astype(float)
    def obj(theta):
        t = float(np.exp(theta[0]))
        return log_loss(y, clip(sigmoid(x/t)), labels=[0,1])
    res = minimize(obj, [0.0], method='Nelder-Mead', options={'maxiter':200})
    return float(np.exp(res.x[0])) if res.success else 1.0

def apply_temp(p, t): return sigmoid(logit(p)/max(float(t), EPS))

def fit_platt(y, p):
    if len(np.unique(y)) < 2: return Platt()
    x = logit(p); y = y.astype(float)
    def obj(theta):
        return log_loss(y, clip(sigmoid(theta[0]*x + theta[1])), labels=[0,1])
    res = minimize(obj, [1.0, 0.0], method='Nelder-Mead', options={'maxiter':400})
    return Platt(float(res.x[0]), float(res.x[1])) if res.success else Platt()

def fit_iso(y, p):
    if len(np.unique(y)) < 2: return None
    return IsotonicRegression(out_of_bounds='clip').fit(clip(p), y.astype(float))

def ece(y, p, n_bins=10, adaptive=False):
    y = np.asarray(y).astype(float); p = clip(p)
    if adaptive:
        q = np.linspace(0,1,n_bins+1)
        edges = np.unique(np.quantile(p, q))
        if len(edges) <= 2: edges = np.linspace(0,1,n_bins+1)
    else:
        edges = np.linspace(0,1,n_bins+1)
    val = 0.0; mce = 0.0
    for i in range(len(edges)-1):
        if i == len(edges)-2:
            idx = (p >= edges[i]) & (p <= edges[i+1])
        else:
            idx = (p >= edges[i]) & (p < edges[i+1])
        if idx.sum() == 0: continue
        gap = abs(float(p[idx].mean()) - float(y[idx].mean()))
        val += idx.mean() * gap
        mce = max(mce, gap)
    return float(val), float(mce)

def metrics_for_matrix(y, p, labels, bins=10):
    rows=[]
    for j, lab in enumerate(labels):
        yy=y[:,j].astype(int); pp=clip(p[:,j])
        if len(np.unique(yy)) < 2:
            auc=np.nan; nll=np.nan
        else:
            auc=float(roc_auc_score(yy,pp)); nll=float(log_loss(yy,pp,labels=[0,1]))
        ew,mce=ece(yy,pp,bins,False); ace,_=ece(yy,pp,bins,True)
        rows.append({'label':lab,'n':int(len(yy)),'positives':int(yy.sum()),'prevalence':float(yy.mean()),
                     'ece':ew,'ace':ace,'mce':mce,'brier':float(brier_score_loss(yy,pp)),'nll':nll,'auc':auc})
    return pd.DataFrame(rows)

def macro_row(dataset, level, method, y, p, labels, bins=10):
    df=metrics_for_matrix(y,p,labels,bins)
    return {'dataset':dataset,'level':level,'method':method,'bins':bins,
            'ece':float(df.ece.mean()),'ace':float(df.ace.mean()),'mce':float(df.mce.mean()),
            'brier':float(df.brier.mean()),'nll':float(df.nll.mean()),'auc':float(df.auc.mean()),
            'n_labels':len(df),'total_n':int(df.n.sum()),'total_pos':int(df.positives.sum())}

def fit_apply_all(y_cal, p_cal, p_test, method):
    out=np.zeros_like(p_test, dtype=float)
    for j in range(p_test.shape[1]):
        if method=='temperature':
            out[:,j]=apply_temp(p_test[:,j], fit_temp(y_cal[:,j], p_cal[:,j]))
        elif method=='platt':
            out[:,j]=fit_platt(y_cal[:,j], p_cal[:,j]).apply(p_test[:,j])
        elif method=='isotonic':
            im=fit_iso(y_cal[:,j], p_cal[:,j])
            out[:,j]=im.predict(clip(p_test[:,j])) if im is not None else p_test[:,j]
    return clip(out)

def load_preds(npz_path):
    z=np.load(npz_path, allow_pickle=True)
    d={'paths':z['sample_paths'], 'child_labels':z['child_labels'].astype(int), 'parent_labels':z['parent_labels'].astype(int)}
    for m in MODEL_NAMES:
        d[f'{m}_child']=clip(z[f'{m}_child_preds'])
        d[f'{m}_parent']=clip(z[f'{m}_parent_preds'])
    d['fused_child']=clip(np.mean([d[f'{m}_child'] for m in MODEL_NAMES], axis=0))
    d['fused_parent']=clip(np.mean([d[f'{m}_parent'] for m in MODEL_NAMES], axis=0))
    return d

def task1(d, outdir, seed):
    cal_idx,test_idx,pids=patient_split(d['paths'],0.5,seed)
    rows=[]
    for member in MODEL_NAMES+['fused']:
        for level, labels in [('child',CHILD_LABELS),('parent',PARENT_LABELS)]:
            y=d[f'{level}_labels']; p=d[f'{member}_{level}']
            methods={'baseline':p[test_idx]}
            for meth in ['temperature','platt','isotonic']:
                methods[meth]=fit_apply_all(y[cal_idx],p[cal_idx],p[test_idx],meth)
            for meth, pp in methods.items():
                rows.append(macro_row('powered_patient_test',level,member+'_'+meth,y[test_idx],pp,labels))
    res=pd.DataFrame(rows)
    res.to_csv(outdir/'task1_single_vs_ensemble_calibrated.csv', index=False)
    child=res[res.level.eq('child')].copy()
    # Decision: keep ensemble in title only if fused is best or near-best (within 0.005 AUC and 0.005 ECE)
    # under the advocated Platt calibration on child labels.
    pl=child[child.method.str.endswith('_platt')].copy()
    pl['member']=pl.method.str.replace('_platt','',regex=False)
    best_auc=pl.auc.max(); best_ece=pl.ece.min()
    fused=pl[pl.member.eq('fused')].iloc[0]
    auc_gap=float(best_auc-fused.auc); ece_gap=float(fused.ece-best_ece)
    keep = (auc_gap <= 0.005) and (ece_gap <= 0.005)
    decision={'task':'Task 1 single-vs-ensemble calibration','rule':'Keep ensemble in title only if fused Platt child AUC is within 0.005 of best member and fused Platt child ECE is within 0.005 of best member.',
              'best_platt_child_auc':float(best_auc),'best_platt_child_ece':float(best_ece),
              'fused_platt_child_auc':float(fused.auc),'fused_platt_child_ece':float(fused.ece),
              'auc_gap_best_minus_fused':auc_gap,'ece_gap_fused_minus_best':ece_gap,
              'outcome':'KEEP_ENSEMBLE_IN_TITLE' if keep else 'DROP_ENSEMBLE_FROM_TITLE_OR_DEMOTE_TO_METHODS'}
    return decision

def task2(d,outdir,seeds,bins_list):
    rows=[]
    for seed in seeds:
        cal_idx,test_idx,pids=patient_split(d['paths'],0.5,seed)
        for level,labels in [('child',CHILD_LABELS),('parent',PARENT_LABELS)]:
            y=d[f'{level}_labels']; p=d[f'fused_{level}']
            calibrated={
                'baseline':p[test_idx],
                'temperature':fit_apply_all(y[cal_idx],p[cal_idx],p[test_idx],'temperature'),
                'platt':fit_apply_all(y[cal_idx],p[cal_idx],p[test_idx],'platt'),
                'isotonic':fit_apply_all(y[cal_idx],p[cal_idx],p[test_idx],'isotonic'),
            }
            for meth,pp in calibrated.items():
                for b in bins_list:
                    r=macro_row('powered_patient_test',level,meth,y[test_idx],pp,labels,bins=b)
                    r['split_seed']=seed
                    rows.append(r)
    df=pd.DataFrame(rows)
    df.to_csv(outdir/'task2_ece_stress_splits_bins.csv', index=False)
    summ=df.groupby(['level','method']).agg(ece_median=('ece','median'),ece_p05=('ece',lambda x: float(np.quantile(x,.05))),ece_p95=('ece',lambda x: float(np.quantile(x,.95))),ece_max=('ece','max'),auc_median=('auc','median'),n=('ece','count')).reset_index()
    summ.to_csv(outdir/'task2_ece_stress_summary.csv', index=False)
    # Trust low ECE if Platt child p95 remains <0.03 and all tested Platt ECE <0.05; isotonic separately flagged as possibly flexible.
    child_platt=summ[(summ.level=='child')&(summ.method=='platt')].iloc[0]
    parent_platt=summ[(summ.level=='parent')&(summ.method=='platt')].iloc[0]
    trust = (child_platt.ece_p95 < 0.03) and (child_platt.ece_max < 0.05) and (parent_platt.ece_p95 < 0.04)
    decision={'task':'Task 2 low-ECE stress test','rule':'Trust low Platt ECE if across patient split seeds and bins 5/10/15/20 child Platt ECE p95 < 0.03, child max < 0.05, and parent Platt ECE p95 < 0.04. Treat isotonic as supportive but more flexible.',
              'child_platt_ece_median':float(child_platt.ece_median),'child_platt_ece_p95':float(child_platt.ece_p95),'child_platt_ece_max':float(child_platt.ece_max),
              'parent_platt_ece_median':float(parent_platt.ece_median),'parent_platt_ece_p95':float(parent_platt.ece_p95),'parent_platt_ece_max':float(parent_platt.ece_max),
              'outcome':'LOW_ECE_PASSES_STRESS_TEST' if trust else 'LOW_ECE_NEEDS_CAUTION_OR_MORE_STRESS_TESTS'}
    return decision

def task3(d,outdir,seed,n_boot=2000):
    cal_idx,test_idx,pids=patient_split(d['paths'],0.5,seed)
    y_child=d['child_labels'][test_idx]; p_child=d['fused_child'][test_idx]
    y_parent=d['parent_labels'][test_idx]; p_parent=d['fused_parent'][test_idx]
    child_df=metrics_for_matrix(y_child,p_child,CHILD_LABELS)
    parent_df=metrics_for_matrix(y_parent,p_parent,PARENT_LABELS)
    child_df['level']='child'; parent_df['level']='parent'
    all_labels=pd.concat([child_df,parent_df], ignore_index=True)
    all_labels.to_csv(outdir/'task3_label_level_metrics_for_matching.csv',index=False)

    parent_prev=parent_df.prevalence.to_numpy()
    child_prev=child_df.prevalence.to_numpy()
    # Match each parent to closest child prevalence without replacement where possible.
    remaining=list(range(len(child_df)))
    pairs=[]
    for i,row in parent_df.sort_values('prevalence').iterrows():
        j=min(remaining, key=lambda k: abs(child_df.loc[k,'prevalence']-row.prevalence))
        remaining.remove(j)
        pairs.append({'parent_label':row.label,'parent_prevalence':row.prevalence,'parent_ece':row.ece,
                      'child_label':child_df.loc[j,'label'],'child_prevalence':child_df.loc[j,'prevalence'],'child_ece':child_df.loc[j,'ece'],
                      'child_minus_parent_ece':child_df.loc[j,'ece']-row.ece})
    pairs_df=pd.DataFrame(pairs)
    pairs_df.to_csv(outdir/'task3_prevalence_matched_parent_child_pairs.csv',index=False)
    observed=float(pairs_df.child_minus_parent_ece.mean())
    rng=np.random.default_rng(seed+333)
    diffs=[]
    for _ in range(n_boot):
        samp=rng.integers(0,len(pairs_df),len(pairs_df))
        diffs.append(float(pairs_df.child_minus_parent_ece.iloc[samp].mean()))
    ci=np.quantile(diffs,[.025,.975])
    # Decision: keep hierarchy gap only if prevalence-matched mean child-parent ECE >0 and 95% CI excludes 0.
    keep = observed > 0 and ci[0] > 0
    decision={'task':'Task 3 prevalence-matched parent/child calibration gap','rule':'Keep parent-vs-child hierarchy gap claim only if prevalence-matched child-minus-parent ECE is positive and bootstrap 95% CI excludes 0.',
              'matched_pairs':int(len(pairs_df)),'mean_child_minus_parent_ece':observed,
              'bootstrap_ci95_low':float(ci[0]),'bootstrap_ci95_high':float(ci[1]),
              'outcome':'KEEP_HIERARCHY_GAP_CLAIM' if keep else 'REMOVE_OR_DEMOTE_HIERARCHY_GAP_CLAIM'}
    return decision

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20260802)
    ap.add_argument('--stress-seeds',default='20260802,20260803,20260804,20260805,20260806,20260807,20260808,20260809,20260810,20260811')
    ap.add_argument('--bins',default='5,10,15,20')
    args=ap.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    d=load_preds(args.npz)
    decisions=[]
    decisions.append(task1(d,args.outdir,args.seed))
    decisions.append(task2(d,args.outdir,[int(x) for x in args.stress_seeds.split(',')],[int(x) for x in args.bins.split(',')]))
    decisions.append(task3(d,args.outdir,args.seed))
    (args.outdir/'tasks1_3_decisions.json').write_text(json.dumps(decisions,indent=2))
    lines=['# Tasks 1-3 hard-gate decision report','']
    for dec in decisions:
        lines += [f"## {dec['task']}", f"Rule: {dec['rule']}", f"Outcome: {dec['outcome']}", '']
        for k,v in dec.items():
            if k not in {'task','rule','outcome'}:
                lines.append(f"- {k}: {v}")
        lines.append('')
    (args.outdir/'tasks1_3_decision_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':
    main()
