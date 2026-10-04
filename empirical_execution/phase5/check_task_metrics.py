#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hashlib,json,warnings
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
from phase5.task_metrics import binary_rank_metrics,civil_metrics,multilabel_metrics,pair_effects

def main():
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    # Algebraic tie/zero/weight fixtures only, not synthetic empirical data.
    y=np.array([[1,0,0],[0,1,0],[1,0,0],[0,1,0],[0,0,0]])
    s=np.array([[.5,.5,-1],[.5,.8,-1],[.8,.5,-1],[.1,.8,-1],[.1,.1,-1]])
    w=np.array([1.,3.,2.,0.,2.]);t=np.array([.5,.5,np.inf])
    result=multilabel_metrics(y,s,t,w)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for average in ('macro','micro'):
            check(average+'_ap_matches_independent_sklearn',abs(result[average+'_average_precision']-average_precision_score(y,s,average=average,sample_weight=w))<1e-14)
            check(average+'_f1_matches_independent_sklearn',abs(result[average+'_f1']-f1_score(y,s>=t,average=average,sample_weight=w,zero_division=0))<1e-14)
    for j in (0,1):
        check('auc_ties_weighted_'+str(j),abs(binary_rank_metrics(y[:,j],s[:,j],w)['auroc']-roc_auc_score(y[:,j],s[:,j],sample_weight=w))<1e-14)
    check('zero_positive_ap_zero_auc_undefined',result['per_output_rank_metrics'][2]['average_precision']==0 and result['per_output_rank_metrics'][2]['auroc'] is None)
    check('zero_target_kept',result['zero_target_rows_retained']==1)
    repeat=np.repeat(np.arange(len(w)),w.astype(int))
    repeated=multilabel_metrics(y[repeat],s[repeat],t)
    check('whole_source_multiplicity_equals_expansion',all(abs(result[k]-repeated[k])<1e-14 for k in ('mse','micro_f1','macro_f1','micro_average_precision','macro_average_precision')))
    check('unclipped_regression',civil_metrics([0,1],[-1,2])['mse']==1.)
    check('constant_predelete_no_floor',pair_effects([0,1],[.5,.5],[.2,.8],[.3,.7])['normalized_prediction_rms'] is None)
    check('negative_signed_loss_preserved',pair_effects([0,1],[.5,.5],[0,1],[.3,.7])['signed_normalized_loss']<0)
    for name,call in [
        ('negative_weights',lambda: civil_metrics([0,1],[0,1],[-1,2])),
        ('fractional_multihot',lambda:multilabel_metrics(y+.1,s,t)),
        ('nan_score',lambda:civil_metrics([0,1],[np.nan,0])),
        ('invalid_threshold',lambda:multilabel_metrics(y,s,[.5,.5,-np.inf]))]:
        try:call()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    # Genuine historical predictions: no new split or tuning.
    old=ROOT/'phase2/results'
    labels=json.loads((old/'evaluation_targets.json').read_text())['original_labels']
    errors=[];count=0
    for line in (old/'predictions.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['dimension']!=64:continue
        for key in ('frozen_predictions','oracle_predictions'):
            metrics=civil_metrics(labels,row[key]);count+=1
            errors.append(abs(metrics['mae']-float(np.mean(np.abs(np.asarray(row[key]).reshape(-1)-np.asarray(labels).reshape(-1))))))
    check('saved_natural_predictions_mae_exact',max(errors,default=0)==0)
    output={'status':'passed','checks':len(checks),'names':checks,'historical_prediction_vectors_checked':count,
            'synthetic_empirical_data':False,'primary_study_result':False,
            'scope':'algebraic metric tests and existing natural Civil18 heldout development predictions',
            'source_sha256':hashlib.sha256(Path(__file__).with_name('task_metrics.py').read_bytes()).hexdigest()}
    dest=ROOT/'phase5/results/task_metrics_checks.json';dest.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
