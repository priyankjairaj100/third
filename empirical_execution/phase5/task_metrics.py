"""Fixed task metrics on immutable evaluation rows; no fitting or clipping.

AP is the noninterpolated threshold-group average precision, not trapezoidal
PR area. Zero-positive AP and zero-denominator F1 use explicit zero conventions;
one-class AUROC is undefined. Weighted rows support whole-source resampling.
"""
import numpy as np
from phase5.statistics import prediction_metrics

def _arrays(y,score,weights=None):
    y=np.asarray(y,dtype=np.float64);s=np.asarray(score,dtype=np.float64)
    if y.ndim==1:y=y[:,None]
    if s.ndim==1:s=s[:,None]
    if y.ndim!=2 or y.shape!=s.shape or not all(y.shape) or not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError('Aligned finite nonempty [row,output] arrays required')
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=np.float64)
    if w.shape!=(len(y),) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Finite nonnegative row weights with positive total required')
    return y,s,w

def binary_rank_metrics(y,score,weights=None):
    y,s,w=_arrays(y,score,weights)
    if y.shape[1]!=1 or not np.isin(y,[0,1]).all():raise ValueError('Binary scalar labels required')
    y=y[:,0];s=s[:,0];active=w>0;y=y[active];s=s[active];w=w[active]
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order];w=w[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(w*y)[ends];fp=np.cumsum(w*(1-y))[ends]
    pos=float(tp[-1]);neg=float(fp[-1])
    ap=float(np.sum(np.diff(np.r_[0.,tp])/pos * (tp/(tp+fp)))) if pos else 0.
    auc=float(np.sum(np.diff(np.r_[0.,fp])*((np.r_[0.,tp[:-1]]+tp)/2))/(pos*neg)) if pos and neg else None
    return {'average_precision':ap,'auroc':auc,'positive_weight':pos,'negative_weight':neg,
            'average_precision_zero_positive_convention':0.,'auroc_defined':auc is not None}

def civil_metrics(y,score,weights=None):
    y,s,w=_arrays(y,score,weights)
    if y.shape[1]!=1 or ((y<0)|(y>1)).any():raise ValueError('Original scalar toxicity fractions in [0,1] required')
    rank=binary_rank_metrics(y>=.5,s,w)
    return {'task':'original_fractional_toxicity','n':len(y),'weight':float(w.sum()),
            'mse':float(np.average((s-y)[:,0]**2,weights=w)),
            'mae':float(np.average(np.abs(s-y)[:,0],weights=w)),
            'auroc_at_toxicity_ge_05':rank['auroc'],
            'average_precision_at_toxicity_ge_05':rank['average_precision'],
            'positive_weight':rank['positive_weight'],'score_clipping':False}

def multilabel_metrics(y,scores,thresholds,weights=None):
    y,s,w=_arrays(y,scores,weights)
    if not np.isin(y,[0,1]).all():raise ValueError('Original multihot targets required')
    t=np.asarray(thresholds,dtype=np.float64)
    if t.shape!=(y.shape[1],) or np.isnan(t).any() or np.isneginf(t).any():
        raise ValueError('One locked threshold per output; +inf means never positive')
    p=s>=t;truth=y==1
    tp=np.sum(w[:,None]*(p&truth),axis=0)
    denom=np.sum(w[:,None]*p,axis=0)+np.sum(w[:,None]*truth,axis=0)
    f1=np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0)
    ranks=[binary_rank_metrics(y[:,j],s[:,j],w) for j in range(y.shape[1])]
    micro=binary_rank_metrics(y.ravel(),s.ravel(),np.repeat(w,y.shape[1]))
    return {'task':'original_multihot_tags','n':len(y),'outputs':y.shape[1],'weight':float(w.sum()),
            'mse':float(np.average(np.mean((s-y)**2,axis=1),weights=w)),
            'sample_averaged_sum_squared_error':float(np.average(np.sum((s-y)**2,axis=1),weights=w)),
            'micro_f1':float(2*tp.sum()/denom.sum()) if denom.sum() else 0.,
            'macro_f1':float(np.mean(f1)),
            'micro_average_precision':micro['average_precision'],
            'macro_average_precision':float(np.mean([r['average_precision'] for r in ranks])),
            'per_output_f1':f1.tolist(),'per_output_rank_metrics':ranks,
            'zero_positive_outputs':sum(r['positive_weight']==0 for r in ranks),
            'zero_target_rows_retained':int(np.sum(~truth.any(axis=1))),
            'zero_positive_AP_convention':0.,'zero_denominator_F1_convention':0.,
            'thresholds_refit':False,'threshold_comparison':'>=','score_clipping':False}

def pair_effects(y,before,frozen,oracle,thresholds=None):
    """Unweighted fixed-test primary effects; signed negative outcomes retained."""
    return prediction_metrics(y,before,frozen,oracle,thresholds)
