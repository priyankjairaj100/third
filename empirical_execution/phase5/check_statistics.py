#!/usr/bin/env python3
"""Statistical algebra checks and compatibility with existing natural outputs."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from phase5.statistics import (lock_registry,reconcile_registry,prediction_metrics,admission_metrics,
    summarize_registry,paired_method_bootstrap,crossed_source_bootstrap,holm_family,exact_activation_interval,equal_rs_lifecycle_difference,equal_rs_lifecycle_bootstrap)


def main():
    tests={}
    cells=[{'dataset_id':'algebra_fixture','panel_id':'software_only','configuration_id':'fixed',
            'arm':'R','trajectory_id':t,'checkpoint':k} for t in ['p0','p1','p2'] for k in [1,2]]
    lock=lock_registry(cells)
    results=[{**cell,'status':'success','metrics':{'a':float(int(cell['trajectory_id'][1:])),
             'scaled':2*int(cell['trajectory_id'][1:]),'any_admission':0.}} for cell in cells]
    out=summarize_registry(lock,results,metrics=['a','scaled','any_admission'])
    tests['whole_path_draws_shared']=out['groups'][0]['metrics']['a']['interval_95']==out['groups'][1]['metrics']['a']['interval_95']
    tests['same_draw_paired_metrics']=np.allclose(np.asarray(out['groups'][0]['metrics']['a']['interval_95'])*2,out['groups'][0]['metrics']['scaled']['interval_95'])
    paired=paired_method_bootstrap(lock,results,method_metric='scaled',baseline_metric='a')
    tests['paired_ratio_recomputed']=paired['groups'][0]['ratio_of_means']['estimate']==2 and paired['groups'][0]['ratio_of_means']['bootstrap_undefined_replicates']>0 and paired['groups'][0]['ratio_of_means']['interval_95'] is None
    tests['paired_differences_shared']=paired['groups'][0]['mean_difference']['interval_95']==out['groups'][0]['metrics']['a']['interval_95']
    tests['all_zero_preserved']=out['groups'][0]['metrics']['any_admission']['mean']==0 and out['groups'][0]['metrics']['any_admission']['observed_zero_values']==3
    partial=summarize_registry(lock,results[:-1],metrics=['a'])
    tests['missing_cell_kept']=partial['groups'][1]['statuses']['missing']==1 and partial['groups'][1]['metrics']['a']['mean'] is None and partial['groups'][1]['metrics']['a']['interval_95'] is None
    try:reconcile_registry(lock,results+[results[0]])
    except ValueError:tests['duplicate_refused']=True
    else:tests['duplicate_refused']=False
    tampered=json.loads(json.dumps(lock));tampered['cells'].pop()
    try:reconcile_registry(tampered,results)
    except ValueError:tests['tampered_lock_refused']=True
    else:tests['tampered_lock_refused']=False
    tests['zero_loss_and_sd_undefined']=prediction_metrics([0,0],[0,0],[0,0],[0,0])['signed_normalized_loss'] is None and prediction_metrics([0,0],[.1,.1],[0,0],[0,0])['normalized_prediction_rms'] is None
    tests['negative_effect_preserved']=prediction_metrics([0,1],[.2,.8],[0,1],[.1,.9])['signed_normalized_loss']<0
    tests['empty_request_ratio_undefined']=admission_metrics(0,0)['admissions_per_deleted_record'] is None
    interval=exact_activation_interval(0,256)
    tests['binomial_zero_event_upper']=abs(interval['one_sided_upper']-(1-.05**(1/256)))<1e-14
    h=holm_family({'civil_comments:R':.001,'civil_comments:S':.02,'askubuntu:R':None,'askubuntu:S':.03},family='signed_loss')
    tests['holm_full_family_missing_retained']=h['family_size']==4 and h['tests']['civil_comments:R']['adjusted_p_value']==.004 and h['tests']['civil_comments:S']['adjusted_p_value']==.06 and not h['tests']['askubuntu:R']['reject']
    try:holm_family({'civil_comments:R':.001},family='signed_loss')
    except ValueError:tests['partial_family_refused']=True
    else:tests['partial_family_refused']=False
    mix=equal_rs_lifecycle_difference([0,0,0],[10])
    tests['arms_equal_not_sample_weighted']=mix['equal_RS_mean']==5
    mixboot=equal_rs_lifecycle_bootstrap([0,0,0],[10])
    tests['equal_arm_bootstrap_independent_and_fixed']=mixboot['interval_95']==[5.,5.] and not mixboot['arms_paired']
    y=np.asarray([[0.],[1.],[1.],[0.]])
    before=np.asarray([[.1],[.1],[.8],[.8]])
    f=np.stack([np.stack([before,before+.1]),np.stack([before-.1,before])])
    o=np.broadcast_to(y,(2,2,4,1)).copy()
    kwargs={'trajectory_ids':['a','b'],'checkpoint_ids':[1,2],'test_source_ids':['g0','g0','g1','g1'],
            'source_kinds':{'g0':'software_fixture_only','g1':'software_fixture_only'},'software_fixture_only':True,'thresholds':[.5]}
    cross=crossed_source_bootstrap(y,before,f,o,**kwargs)
    tests['crossed_recomputes_zero_denominators']=cross['undefined_denominator_replicates']['predelete_score_sd_zero']>0 and cross['groups'][0]['metrics']['normalized_prediction_rms']['interval_95'] is None
    tests['macro_f1_recomputed']=cross['nonlinear_macro_f1_recomputed'] and cross['groups'][0]['metrics']['macro_f1_difference']['defined_replicates']==10_000
    unknown={**kwargs,'test_source_ids':['u0','u1','u2','u3'],'source_kinds':{f'u{i}':'unknown_singleton' for i in range(4)},'software_fixture_only':False}
    try:crossed_source_bootstrap(y,before,f,o,**unknown)
    except ValueError:tests['unknown_sources_not_native_substitute']=True
    else:tests['unknown_sources_not_native_substitute']=False
    # Reconstruct every Phase 2 d64 metric from actual saved predictions, preserve
    # its original diagnostic split and leakage limitations, no new experiment.
    old=ROOT/'phase2/results'
    labels=json.loads((old/'evaluation_targets.json').read_text())['original_labels']
    utility=next(r for r in json.loads((old/'utility_context.json').read_text()) if r['dimension']==64)
    checkpoints=[json.loads(l) for l in (old/'checkpoints.jsonl').read_text().splitlines()]
    checkpoints={(r['trajectory_id'],r['checkpoint']):r for r in checkpoints if r['dimension']==64}
    predictions=[json.loads(l) for l in (old/'predictions.jsonl').read_text().splitlines()]
    actual=[];plan=[];maxdiff=0.
    for r in predictions:
        if r['dimension']!=64:continue
        c=checkpoints[r['trajectory_id'],r['checkpoint']]
        cell={'dataset_id':'civil_comments_reused_preview','panel_id':'phase2_existing_82_18_diagnostic',
              'configuration_id':'frozen_lexical_d64','arm':r['arm'],'trajectory_id':r['trajectory_id'],'checkpoint':r['checkpoint']}
        m=prediction_metrics(labels,utility['predelete_predictions'],r['frozen_predictions'],r['oracle_predictions'])
        old_metric=c.get('metrics',c)
        if 'relative_mse_effect' in old_metric:maxdiff=max(maxdiff,abs(m['signed_normalized_loss']-old_metric['relative_mse_effect']))
        m.update(admission_metrics(c['addition_count'],len(c['deleted_ids'])))
        plan.append(cell);actual.append({**cell,'status':'success','metrics':m})
    registry=lock_registry(plan)
    analysis=summarize_registry(registry,actual,metrics=['signed_normalized_loss','prediction_rms','normalized_prediction_rms','any_admission','admissions'])
    tests['natural_registry_all_rows_retained']=sum(len(g['trajectory_ids']) for g in analysis['groups'])==len(actual)==448
    tests['natural_reconstructed_metric_match']=maxdiff<1e-12
    result={'schema':'ccu-statistical-checks-1','status':'passed' if all(tests.values()) else 'failed',
            'checks':tests,'passed':sum(tests.values()),'total':len(tests),
            'natural_existing_checkpoints':len(actual),'natural_maximum_metric_difference':maxdiff,
            'natural_source_bootstrap_executed':False,'natural_scope':'historical Phase2 lexical diagnostic prediction reanalysis only; known leakage retained',
            'registry_lock_is_retrospective_software_test':True,'primary_study':False,
            'source_sha256':hashlib.sha256((Path(__file__).with_name('statistics.py')).read_bytes()).hexdigest(),
            'check_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    outdir=ROOT/'phase5/results'
    (outdir/'statistics_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    (outdir/'statistics_natural_compatibility.json').write_text(json.dumps(analysis,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert all(tests.values())

if __name__=='__main__':main()
