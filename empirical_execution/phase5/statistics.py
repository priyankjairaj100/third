"""Locked trajectory analysis with preserved failures and source uncertainty.

Percentile bootstrap intervals are descriptive sampling intervals conditional on
stated units. They are not p-values, proof of equivalence, or corpus replication.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import defaultdict
from numbers import Integral
import numpy as np
from scipy.stats import beta

BOOTSTRAP_REPLICATES = 10_000
_FIELDS = ('dataset_id', 'panel_id', 'configuration_id', 'arm', 'trajectory_id', 'checkpoint')
_STATUSES = {'success', 'failed', 'infeasible', 'timeout', 'skipped', 'missing'}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _seal(obj):
    obj = dict(obj)
    obj['sha256'] = hashlib.sha256(_json(obj)).hexdigest()
    return obj


def _verify(obj):
    copy = dict(obj)
    claimed = copy.pop('sha256', None)
    if claimed != hashlib.sha256(_json(copy)).hexdigest():
        raise ValueError('registry checksum mismatch')


def _integer(value, name, lower=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < lower:
        raise ValueError(f'{name} requires integer >= {lower}')
    return int(value)


def _key(row):
    if any(not isinstance(row.get(k), str) or not row[k] for k in _FIELDS[:-1]):
        raise ValueError('complete nonempty analysis identity required')
    return tuple(row[k] for k in _FIELDS[:-1]) + (_integer(row.get('checkpoint'), 'checkpoint'),)


def lock_registry(cells, *, seed=20271004, replicates=BOOTSTRAP_REPLICATES):
    """Create outcome-free cell lock; timestamp/authenticity is external evidence."""
    seed = _integer(seed, 'seed')
    replicates = _integer(replicates, 'replicates', 1)
    cells = list(cells)
    if not cells or any(set(row) != set(_FIELDS) for row in cells):
        raise ValueError('registry contains only complete identity fields, never outcomes')
    keys = [_key(row) for row in cells]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate planned analysis cells')
    groups = defaultdict(lambda: defaultdict(set))
    for key in keys:
        groups[key[:4]][key[4]].add(key[5])
    for paths in groups.values():
        if len({tuple(sorted(v)) for v in paths.values()}) != 1:
            raise ValueError('each trajectory in an analysis group must have the same planned checkpoints')
    return _seal({'schema': 'ccu-trajectory-registry-1', 'seed': seed,
                  'replicates': replicates, 'protocol_replicates': replicates == BOOTSTRAP_REPLICATES,
                  'cells': [dict(zip(_FIELDS, key)) for key in sorted(keys)],
                  'external_preoutcome_timing_verified': False})


def reconcile_registry(registry, records):
    """Return every planned cell; absence stays missing, extras/duplicates fail."""
    _verify(registry)
    planned = {_key(row): row for row in registry['cells']}
    supplied = {}
    for raw in records:
        row = dict(raw)
        key = _key(row)
        if key not in planned or key in supplied:
            raise ValueError('unplanned or duplicated result cell')
        if row.get('status') not in _STATUSES or not isinstance(row.get('metrics'), dict):
            raise ValueError('explicit status and metrics mapping required')
        for name, value in row['metrics'].items():
            if not isinstance(name, str) or not name:
                raise ValueError('metric names must be nonempty strings')
            if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (float,int,np.floating,np.integer)) or not math.isfinite(value)):
                raise ValueError('finite measurements or explicit null required')
        supplied[key] = row
    return [supplied.get(key, {**planned[key], 'status':'missing', 'metrics':{},
                              'failure_reason':'planned cell not supplied'}) for key in sorted(planned)]


def prediction_metrics(y, before, frozen, oracle, thresholds=None):
    """Protocol D, s0, Z and signed/exceedance metrics; no denominator floors."""
    values = []
    for value in (y,before,frozen,oracle):
        a = np.asarray(value,dtype=np.float64)
        if a.ndim == 1: a = a[:,None]
        if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
            raise ValueError('finite nonempty test predictions/labels required')
        values.append(a)
    y,before,frozen,oracle = values
    if len({a.shape for a in values}) != 1:
        raise ValueError('test matrices must have identical shape')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        base = float(np.mean((before-y)**2))
        fm,om = float(np.mean((frozen-y)**2)),float(np.mean((oracle-y)**2))
        raw = float(np.sqrt(np.mean((frozen-oracle)**2)))
        shifted = before-before[:1]
        s0 = float(np.sqrt(np.mean((shifted-shifted.mean(axis=0))**2)))
        signed = (fm-om)/base if base else None
        norm = raw/s0 if s0 else None
    out = {'predelete_mse':base,'predelete_score_sd':s0,'frozen_mse':fm,'oracle_mse':om,
           'signed_raw_loss':fm-om,'signed_normalized_loss':signed,
           'absolute_normalized_loss':abs(signed) if signed is not None else None,
           'prediction_rms':raw,'normalized_prediction_rms':norm,
           'loss_exceedance_001':float(abs(signed)>.01) if signed is not None else None,
           'prediction_exceedance_001':float(norm>.01) if norm is not None else None}
    if thresholds is not None:
        t = np.asarray(thresholds,dtype=float)
        if t.shape != (y.shape[1],) or np.isnan(t).any() or not np.isin(y,[0,1]).all():
            raise ValueError('locked per-output thresholds and binary labels required for macro-F1')
        for name,pred in [('frozen',frozen),('oracle',oracle)]:
            positive=pred>=t
            tp=np.sum(positive & (y==1),axis=0)
            denom=np.sum(positive,axis=0)+np.sum(y==1,axis=0)
            out[name+'_macro_f1']=float(np.mean(np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom!=0)))
        out['macro_f1_difference']=out['frozen_macro_f1']-out['oracle_macro_f1']
    return out


def admission_metrics(admissions, deleted_records, deleted_sources=None):
    count=_integer(admissions,'admissions');nr=_integer(deleted_records,'deleted_records')
    out={'admissions':float(count),'any_admission':float(count>0),
         'admissions_per_deleted_record':count/nr if nr else None}
    if deleted_sources is not None:
        ns=_integer(deleted_sources,'deleted_sources')
        out['admissions_per_deleted_source']=count/ns if ns else None
    return out


def _summary(values, boot):
    good=np.isfinite(values);defined=np.isfinite(boot)
    all_defined=bool(good.all())
    return {'planned_trajectories':len(values),'defined_values':int(good.sum()),
            'undefined_values':int((~good).sum()),
            'mean':float(values.mean()) if all_defined else None,
            'observed_only_mean_descriptive':float(values[good].mean()) if good.any() else None,
            'full_registry_estimand_identified':all_defined,
            'interval_95':np.quantile(boot,[.025,.975]).tolist() if all_defined and defined.all() else None,
            'bootstrap_defined_replicates':int(defined.sum()),
            'bootstrap_undefined_replicates':int((~defined).sum()),
            'observed_zero_values':int(np.count_nonzero(values==0)),
            'values_in_registry_order':[float(v) if math.isfinite(v) else None for v in values]}


def exact_activation_interval(successes, trials, confidence=.95):
    successes=_integer(successes,'successes');trials=_integer(trials,'trials',1)
    if successes>trials or not 0<confidence<1:raise ValueError('invalid binomial parameters')
    alpha=1-confidence
    lo=0. if successes==0 else float(beta.ppf(alpha/2,successes,trials-successes+1))
    hi=1. if successes==trials else float(beta.ppf(1-alpha/2,successes+1,trials-successes))
    upper=1. if successes==trials else float(beta.ppf(confidence,successes+1,trials-successes))
    return {'two_sided_interval':[lo,hi],'one_sided_upper':upper,'confidence':confidence,
            'condition':'independent identically distributed Bernoulli trajectory outcomes at this checkpoint'}


def summarize_registry(registry, records, *, metrics):
    rows=reconcile_registry(registry,records)
    if not metrics or len(set(metrics))!=len(metrics):raise ValueError('unique metric names required')
    grouped=defaultdict(list)
    for row in rows:grouped[_key(row)[:4]].append(row)
    output=[]
    for group,group_rows in sorted(grouped.items()):
        paths=sorted({r['trajectory_id'] for r in group_rows});points=sorted({r['checkpoint'] for r in group_rows})
        lookup={(r['trajectory_id'],r['checkpoint']):r for r in group_rows}
        array=np.full((len(paths),len(points),len(metrics)),np.nan)
        for ti,t in enumerate(paths):
            for ki,k in enumerate(points):
                for mi,m in enumerate(metrics):
                    value=lookup[t,k]['metrics'].get(m)
                    if value is not None:array[ti,ki,mi]=value
        seed=int.from_bytes(hashlib.sha256(_json([registry['seed'],*group])).digest()[:16],'big')
        rng=np.random.default_rng(seed)
        boot=np.empty((registry['replicates'],len(points),len(metrics)))
        for start in range(0,registry['replicates'],128):
            end=min(registry['replicates'],start+128)
            idx=rng.integers(0,len(paths),size=(end-start,len(paths)))
            # Ordinary mean deliberately propagates any sampled missing value.
            # The SAME whole-path draw serves every checkpoint and metric.
            boot[start:end]=array[idx].mean(axis=1)
        for ki,k in enumerate(points):
            result={'identity':dict(zip(_FIELDS[:4],group)),'checkpoint':k,
                    'trajectory_ids':paths,'bootstrap_group_seed':str(seed),
                    'statuses':{s:sum(lookup[t,k]['status']==s for t in paths) for s in sorted(_STATUSES)},
                    'metrics':{m:_summary(array[:,ki,mi],boot[:,ki,mi]) for mi,m in enumerate(metrics)}}
            if 'any_admission' in metrics:
                v=array[:,ki,metrics.index('any_admission')]
                if np.isfinite(v).all() and np.isin(v,[0,1]).all():
                    result['activation_binomial_interval']=exact_activation_interval(int(v.sum()),len(v))
            output.append(result)
    return {'schema':'ccu-trajectory-analysis-1','registry_sha256':registry['sha256'],
            'bootstrap_replicates':registry['replicates'],'protocol_replicates':registry['replicates']==BOOTSTRAP_REPLICATES,
            'conditioning':'fixed corpus, curator, test set and separate request arm',
            'checkpoint_draws_shared':True,'corpora_or_arms_pooled':False,
            'failure_or_zero_rows_dropped':False,'intervals_are_p_values':False,'groups':output}


def crossed_source_bootstrap(y,before,frozen,oracle,*,trajectory_ids,checkpoint_ids,
        test_source_ids,source_kinds,seed=20271004,replicates=BOOTSTRAP_REPLICATES,thresholds=None,
        software_fixture_only=False):
    """Secondary crossed whole-path x whole-test-source bootstrap.

    Group multiplicities preserve all records of each sampled source. Test-source
    uncertainty and deletion-trajectory uncertainty are separate from training
    corpus/curator uncertainty. Source declarations do not authenticate identity.
    This complete-prediction API refuses nonfinite/missing values; preserve failed
    trajectories in the registry and report the crossed estimand unavailable.
    """
    seed=_integer(seed,'seed');replicates=_integer(replicates,'replicates',1)
    y=np.asarray(y,dtype=float);before=np.asarray(before,dtype=float)
    if y.ndim==1:y=y[:,None]
    if before.ndim==1:before=before[:,None]
    f=np.asarray(frozen,dtype=float);o=np.asarray(oracle,dtype=float)
    if f.ndim!=4 or f.shape!=o.shape or y.ndim!=2 or y.shape!=before.shape or f.shape[2:]!=y.shape or not all(f.shape):
        raise ValueError('predictions must have [trajectory,checkpoint,test,output] shape')
    if not all(np.isfinite(a).all() for a in (y,before,f,o)):raise ValueError('crossed bootstrap unavailable with missing/failing predictions')
    nt,nk,ne,nc=f.shape
    if len(trajectory_ids)!=nt or len(set(trajectory_ids))!=nt or len(checkpoint_ids)!=nk or len(set(checkpoint_ids))!=nk:
        raise ValueError('unique aligned trajectory/checkpoint IDs required')
    if len(test_source_ids)!=ne or any(not isinstance(s,str) or not s for s in test_source_ids):raise ValueError('complete test provenance grouping required')
    groups=sorted(set(test_source_ids));ng=len(groups)
    if set(source_kinds)!=set(groups) or any(k not in {'genuine_native','unknown_singleton','software_fixture_only'} for k in source_kinds.values()):raise ValueError('complete declared source kinds required')
    if any(k=='software_fixture_only' for k in source_kinds.values()) and not software_fixture_only:raise ValueError('fixture source kinds require explicit software-only role')
    counts={s:test_source_ids.count(s) if hasattr(test_source_ids,'count') else sum(t==s for t in test_source_ids) for s in groups}
    if any(source_kinds[s]=='unknown_singleton' and counts[s]!=1 for s in groups):raise ValueError('unknown sources must be singleton groups')
    if not software_fixture_only and sum(k=='genuine_native' for k in source_kinds.values())<2:raise ValueError('at least two genuine test sources required; no fake source substitutes')
    gi=np.asarray([groups.index(s) for s in test_source_ids])
    size=np.asarray([counts[s] for s in groups],dtype=float)
    sq0=np.asarray([np.sum((before[gi==g]-y[gi==g])**2) for g in range(ng)])
    group_means=[];within_ss=[]
    for g in range(ng):
        values=before[gi==g];shifted=values-values[:1]
        center=shifted.mean(axis=0)
        group_means.append(values[0]+center)
        within_ss.append(np.sum((shifted-center)**2))
    group_means=np.asarray(group_means);within_ss=np.asarray(within_ss)
    loss=np.stack([np.sum((f[:,:,gi==g,:]-y[gi==g])**2-(o[:,:,gi==g,:]-y[gi==g])**2,axis=(2,3)) for g in range(ng)],axis=-1)
    diff=np.stack([np.sum((f[:,:,gi==g,:]-o[:,:,gi==g,:])**2,axis=(2,3)) for g in range(ng)],axis=-1)
    confusion=None
    if thresholds is not None:
        t=np.asarray(thresholds,dtype=float)
        if t.shape!=(nc,) or np.isnan(t).any() or not np.isin(y,[0,1]).all():raise ValueError('locked per-output thresholds and binary labels required')
        confusion=[]
        for pred in (f,o):
            positive=pred>=t
            tp=np.stack([np.sum(positive[:,:,gi==g,:] & (y[gi==g]==1),axis=2) for g in range(ng)],axis=-1)
            den=np.stack([np.sum(positive[:,:,gi==g,:],axis=2)+np.sum(y[gi==g]==1,axis=0) for g in range(ng)],axis=-1)
            confusion.append((tp,den))
    names=['signed_normalized_loss','prediction_rms','normalized_prediction_rms']
    if confusion is not None:names+=['macro_f1_difference']
    boot=np.full((replicates,nk,len(names)),np.nan)
    rng=np.random.default_rng(seed)
    undefined_denominators={'predelete_mse_zero':0,'predelete_score_sd_zero':0}
    for b in range(replicates):
        tw=rng.multinomial(nt,np.full(nt,1/nt))/nt
        sw=rng.multinomial(ng,np.full(ng,1/ng))
        nr=float(size@sw);mse=float(sq0@sw)/(nr*nc)
        shifted=group_means-group_means[np.flatnonzero(sw)[0]]
        center=(shifted.T@(size*sw))/nr
        var=float((within_ss@sw + np.sum((shifted-center)**2*(size*sw)[:,None]))/(nr*nc))
        sd=float(np.sqrt(var))
        raw=np.sqrt(np.maximum(0.,diff@sw/(nr*nc)))
        if mse:boot[b,:,0]=tw@(loss@sw/(nr*nc*mse))
        else:undefined_denominators['predelete_mse_zero']+=1
        boot[b,:,1]=tw@raw
        if sd:boot[b,:,2]=tw@(raw/sd)
        else:undefined_denominators['predelete_score_sd_zero']+=1
        if confusion is not None:
            vals=[]
            for tp,den in confusion:
                a,c=tp@sw,den@sw
                vals.append(np.mean(np.divide(2*a,c,out=np.zeros_like(a,dtype=float),where=c!=0),axis=2))
            boot[b,:,3]=tw@(vals[0]-vals[1])
    result=[]
    for k,checkpoint in enumerate(checkpoint_ids):
        metrics={}
        for j,name in enumerate(names):
            v=boot[:,k,j];ok=np.isfinite(v)
            metrics[name]={'interval_95':np.quantile(v,[.025,.975]).tolist() if ok.all() else None,
                           'defined_replicates':int(ok.sum()),'undefined_replicates':int((~ok).sum())}
        result.append({'checkpoint':checkpoint,'metrics':metrics})
    return {'schema':'ccu-crossed-source-bootstrap-1','role':'software_fixture_only' if software_fixture_only else 'secondary_source_sampling_uncertainty',
            'replicates':replicates,'protocol_replicates':replicates==BOOTSTRAP_REPLICATES,'seed':seed,
            'source_groups':ng,'genuine_source_groups':sum(k=='genuine_native' for k in source_kinds.values()),
            'unknown_singleton_records':sum(source_kinds[s]=='unknown_singleton' for s in test_source_ids),
            'source_identity_authenticated_here':False,'trajectory_ids':list(trajectory_ids),
            'denominators_recomputed':True,'nonlinear_macro_f1_recomputed':confusion is not None,
            'whole_checkpoint_paths_preserved':True,'undefined_denominator_replicates':undefined_denominators,
            'training_corpus_uncertainty_covered':False,'groups':result}


def holm_family(p_values, *, family, alpha=.05):
    """Correct externally justified p-values; never derive from percentile CIs.

    Missing tests count in the predeclared family and receive conservative p=1.
    """
    families={'signed_loss':('civil_comments:R','civil_comments:S','askubuntu:R','askubuntu:S'),
              'equal_RS_lifecycle':('civil_comments','askubuntu')}
    if family not in families or set(p_values)!=set(families[family]) or not 0<alpha<1:
        raise ValueError('provide the complete fixed family with null for missing tests')
    effective={}
    for key,p in p_values.items():
        if p is not None and (isinstance(p,bool) or not isinstance(p,(float,int)) or not math.isfinite(p) or not 0<=p<=1):raise ValueError('invalid p-value')
        effective[key]=1. if p is None else p
    ordered=sorted(effective,key=lambda k:(effective[k],k));running=0.;out={};m=len(ordered)
    for i,key in enumerate(ordered):
        running=max(running,min(1.,(m-i)*effective[key]))
        out[key]={'p_value':p_values[key],'adjusted_p_value':running,
                  'reject':p_values[key] is not None and running<=alpha,
                  'missing_test_kept_in_family':p_values[key] is None}
    return {'family':family,'alpha':alpha,'family_size':m,'tests':out,
            'validity_requires_valid_input_p_values':True}


def equal_rs_lifecycle_difference(r_differences,s_differences):
    """Fixed equal-arm mean; no pairing across independently sampled arms."""
    arrays=[np.asarray(x,dtype=float) for x in (r_differences,s_differences)]
    if any(a.ndim!=1 or not len(a) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('complete paired within-trajectory R and S differences required')
    return {'R_mean':float(arrays[0].mean()),'S_mean':float(arrays[1].mean()),
            'equal_RS_mean':float(.5*arrays[0].mean()+.5*arrays[1].mean()),
            'R_trajectories':len(arrays[0]),'S_trajectories':len(arrays[1]),
            'deployment_mixture_claim':False}


def paired_method_bootstrap(registry, records, *, method_metric, baseline_metric):
    """Paired mean differences and ratio of means, not mean of row ratios."""
    rows=reconcile_registry(registry,records)
    grouped=defaultdict(list)
    for row in rows:grouped[_key(row)[:4]].append(row)
    out=[]
    for group,values in sorted(grouped.items()):
        paths=sorted({r['trajectory_id'] for r in values});points=sorted({r['checkpoint'] for r in values})
        lookup={(r['trajectory_id'],r['checkpoint']):r for r in values}
        a=np.full((len(paths),len(points),2),np.nan)
        for ti,t in enumerate(paths):
            for ki,k in enumerate(points):
                for mi,m in enumerate([method_metric,baseline_metric]):
                    value=lookup[t,k]['metrics'].get(m)
                    if value is not None:a[ti,ki,mi]=value
        seed=int.from_bytes(hashlib.sha256(_json([registry['seed'],*group])).digest()[:16],'big')
        rng=np.random.default_rng(seed)
        mean_boot=np.empty((registry['replicates'],len(points),2))
        for start in range(0,registry['replicates'],128):
            end=min(registry['replicates'],start+128)
            draw=rng.integers(0,len(paths),size=(end-start,len(paths)))
            mean_boot[start:end]=a[draw].mean(axis=1)
        for ki,k in enumerate(points):
            am,bm=mean_boot[:,ki,0],mean_boot[:,ki,1]
            ratio=np.full(len(am),np.nan)
            valid=np.isfinite(am)&np.isfinite(bm)&(bm!=0)
            ratio[valid]=am[valid]/bm[valid]
            av,bv=a[:,ki,0],a[:,ki,1]
            defined=np.isfinite(av).all() and np.isfinite(bv).all() and bv.mean()!=0
            out.append({'identity':dict(zip(_FIELDS[:4],group)),'checkpoint':k,
                'trajectory_ids':paths,'method_metric':method_metric,'baseline_metric':baseline_metric,
                'mean_difference':_summary(av-bv,am-bm),
                'ratio_of_means':{'estimate':float(av.mean()/bv.mean()) if defined else None,
                    'interval_95':np.quantile(ratio,[.025,.975]).tolist() if defined and valid.all() else None,
                    'bootstrap_defined_replicates':int(valid.sum()),
                    'bootstrap_undefined_replicates':int((~valid).sum()),
                    'full_registry_estimand_identified':bool(defined)}})
    return {'schema':'ccu-paired-method-bootstrap-1','registry_sha256':registry['sha256'],
            'bootstrap_replicates':registry['replicates'],'checkpoint_draws_shared':True,
            'ratio_estimand':'mean(method)/mean(baseline); zero mean baseline undefined',
            'groups':out}


def equal_rs_lifecycle_bootstrap(r_differences,s_differences,*,seed=20271004,
                                replicates=BOOTSTRAP_REPLICATES):
    """Secondary systems-family statistic, independent arm resampling.

    Inputs must be full registry-aligned paired method-minus-baseline differences.
    This helper rejects missing/nonfinite outcomes, never filters them.
    """
    result=equal_rs_lifecycle_difference(r_differences,s_differences)
    seed=_integer(seed,'seed');replicates=_integer(replicates,'replicates',1)
    r,s=np.asarray(r_differences,dtype=float),np.asarray(s_differences,dtype=float)
    # SeedSequence children make independent arm streams; arm sample counts do
    # not change the fixed one-half weighting in the corpus-level estimand.
    streams=np.random.SeedSequence(seed).spawn(2)
    rr,ss=np.random.default_rng(streams[0]),np.random.default_rng(streams[1])
    boot=np.empty(replicates)
    for lo in range(0,replicates,128):
        hi=min(replicates,lo+128)
        ri=rr.integers(0,len(r),size=(hi-lo,len(r)))
        si=ss.integers(0,len(s),size=(hi-lo,len(s)))
        boot[lo:hi]=.5*r[ri].mean(axis=1)+.5*s[si].mean(axis=1)
    return {**result,'interval_95':np.quantile(boot,[.025,.975]).tolist(),
            'bootstrap_replicates':replicates,'seed':seed,'arms_paired':False,
            'p_value_generated':False,'requires_complete_locked_registry_arrays':True}
