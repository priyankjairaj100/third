"""Independent arithmetic/metadata review of actual exported task locks and audit."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def binding(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def load(path,sealed=False):
    value=json.loads(path.read_bytes())
    if sealed:assert value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'seal:'+path.name
    return value


def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps([a.dtype.str,list(a.shape)],separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()


def review(folder,repo):
    task=load(folder/'task_result.json');vocab=load(folder/'vocabulary.json',True);ridge=load(folder/'ridge.json',True);thresholds=load(folder/'thresholds.json',True)
    assert task['passed'] and task['job_id']=='15243' and task['calibration_rows']==20996
    assert task['package_sha256']==hashlib.sha256((repo/'local_run/task_calibration_v3_20261016/package_manifest.json').read_bytes()).hexdigest()
    for n,b in task['outputs'].items():assert binding(folder/n)==b
    for k,n in [('vocabulary_seal','vocabulary.json'),('ridge_seal','ridge.json'),('thresholds_seal','thresholds.json')]:assert task[k]==load(folder/n)['sha256']
    assert all(x['provenance']==vocab['provenance'] for x in (ridge,thresholds))
    assert vocab['provenance']['evidence_role']=='engineering_nonconfirmatory'
    assert ridge['vocabulary_lock_sha256']==vocab['sha256']==thresholds['vocabulary_lock_sha256']
    assert thresholds['lambda_lock_sha256']==ridge['sha256']
    ranked=vocab['ranking_ledger'];assert ranked==sorted(ranked,key=lambda x:(-x['questions'],x['tag']))
    assert vocab['vocabulary']==[x['tag'] for x in ranked[:20]] and len({x['tag'] for x in ranked})==len(ranked)
    assert all(x['selected']==(i<20) for i,x in enumerate(ranked))
    ids=ridge['record_ids'];sources=ridge['source_ids'];folds=ridge['record_folds']
    assert ids==sorted(set(ids))==vocab['calibration_record_ids'] and len(ids)==len(sources)==len(folds)==20996
    unique=sorted(set(sources),key=lambda s:(hashlib.sha256(('ccu-v1-task-cv\0'+s).encode()).digest(),s))
    allocation={s:i%5 for i,s in enumerate(unique)}
    assert folds==[allocation[s] for s in sources]
    assert ridge['fold_ledger']==[{'source_unit_id':s,'fold':allocation[s]} for s in sorted(unique)]
    assert Counter(folds).keys()==set(range(5))
    assert sum(s.startswith('unknown:') for s in unique)==247 and len(unique)==13394
    assert ridge['source_sha256']==digest(list(zip(ids,sources)))
    assert ridge['grid']==[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    ledger=ridge['selection_ledger'];assert [x['lambda'] for x in ledger]==ridge['grid']
    chosen=min(ledger,key=lambda x:(x['mean_squared_loss'],-x['lambda']))
    assert chosen['lambda']==ridge['lambda']==task['lambda']==1e-5
    for entry in ledger:
        assert len(entry['folds'])==5
        for j,f in enumerate(entry['folds']):
            assert f['fold']==j and f['validation_rows']==folds.count(j) and f['training_rows']==20996-folds.count(j)
            assert math.isfinite(f['normalized_solve_residual']) and f['normalized_solve_residual']<=1e-8
    scores=np.asarray(ridge['selected_oof_scores'],np.float64);targets=np.asarray(ridge['calibration_targets'],np.float64)
    assert scores.shape==targets.shape==(20996,20) and np.isfinite(scores).all() and np.all((targets==0)|(targets==1))
    assert array_hash(scores)==ridge['selected_oof_sha256'] and array_hash(targets)==ridge['label_sha256']
    mse=math.fsum(float(x) for x in ((scores-targets)**2).ravel())/targets.size
    assert mse==chosen['mean_squared_loss'] and mse.hex()==chosen['mean_squared_loss_hex']
    assert int(np.sum(np.sum(targets,axis=1)==0))==vocab['all_zero_target_questions']==8032
    for j in range(20):assert int(targets[:,j].sum())==ranked[j]['questions']
    # Independent vectorized tie-block sweep; exact integer cross-products for F1.
    for j,decision in enumerate(thresholds['thresholds']):
        assert decision['output_coordinate']==j
        values,inverse=np.unique(scores[:,j],return_inverse=True)
        counts=np.bincount(inverse);positive_counts=np.bincount(inverse,weights=targets[:,j]).astype(np.int64)
        predicted=np.cumsum(counts[::-1]);tp=np.cumsum(positive_counts[::-1]);positives=int(targets[:,j].sum())
        assert decision['candidate_count']==len(values)+1
        if not positives:
            assert decision['kind']=='never_positive' and decision['threshold'] is None;continue
        best=0
        for i in range(1,len(values)):
            if int(tp[i])*(positives+int(predicted[best]))>int(tp[best])*(positives+int(predicted[i])):best=i
        expected=float(values[::-1][best]);numerator=2*int(tp[best]);denominator=positives+int(predicted[best]);gcd=math.gcd(numerator,denominator)
        assert decision['kind']=='score_at_least' and decision['threshold']==expected and decision['threshold_hex']==expected.hex()
        assert decision['positives']==positives and decision['predicted_positives']==int(predicted[best]) and decision['true_positives']==int(tp[best])
        assert decision['f1']=={'numerator':numerator//gcd,'denominator':denominator//gcd}
    process=load(folder/'task_process.json');tests=load(folder/'task_tests.json')
    assert process['returncode']==0 and not process['console_cap_exceeded'] and process['job_id']=='15243'
    assert tests=={'errors':0,'failures':0,'passed':True,'skipped':0,'tests':8}
    audit=load(folder/'audit_result.json',True);bridge=load(folder/'audit_bridge.json',True)
    assert audit['passed'] and audit['model_alias']=='qwen' and audit['audit_repair']['audit_job_id']=='15241'
    assert audit['audit_repair']['auditor_source']==binding(repo/'local_run/llm_v4_audit_repair_v1_20261015/audit.py')
    assert audit['audit_repair']['repair_package_sha256']==hashlib.sha256((repo/'local_run/llm_v4_audit_repair_v1_20261015/package_manifest.json').read_bytes()).hexdigest()
    assert bridge['source_result']==binding(folder/'audit_result.json') and bridge['source_process_completion']==binding(folder/'audit_process.json')
    assert len(audit['canonical_observations'])==360 and len(audit['diagnostic_observations'])==36 and len(audit['raw_replay_results'])==396
    assert len({x['call_id'] for x in audit['raw_replay_results']})==396
    assert dict(Counter(x['status'] for x in audit['raw_replay_results']))==audit['status_counts']=={'valid':391,'invalid_json':5}
    assert audit['all_call_identities_denominators_and_tokens_verified'] and not audit['semantic_validity_established']
    assert not bridge['original_failed_job_relabelled'] and bridge['model_calls_repeated']==0
    assert bridge['preserved_original_stage_files']['child.stderr.txt']['sha256']=='f23819738ab590c58396c8a27a1d9a1d4d330e3419d234ca889a8114af17a9b3'
    return {'schema':'third-task-lock-and-Qwen-audit-local-review-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,
        'task_job':'15243','audit_job':'15241','reviewer_kind':'AI assistant local arithmetic and metadata review','human_review_claimed':False,
        'bindings':{n:binding(folder/n) for n in ['vocabulary.json','ridge.json','thresholds.json','task_result.json','task_process.json','task_tests.json','audit_result.json','audit_bridge.json','audit_process.json']},
        'verified':{'task_lock_seals':3,'source_group_folds':5,'native_output_coordinates':20,'lambda_candidates':6,'selected_lambda':ridge['lambda'],'selected_OOF_MSE_recomputed_exactly':True,'per_tag_F1_thresholds_independently_recomputed':20,'unknown_singletons_retained':247,'all_zero_target_records_retained':8032,'Qwen_audit_and_bridge_seals':True,'Qwen_output_denominator':396,'original_failed_audit_preserved':True,'zero_model_retries':True},
        'limits':['Source bytes and feature payloads were rehashed by the executed cluster task, not independently reread by this local arithmetic reviewer.','All30 ridge solves were executed by the frozen selector; this review checks their ledgers, selected predictions and exact selection/threshold arithmetic, not a second fit.','Qwen raw-token and grammar replay was executed by the CPU audit; this local review verifies exported audit metadata, bindings, counts and source identity.','No semantic validation, held-out utility result, main effect study or confirmatory acceptance is established.'],
        'source_sha256':binding(Path(__file__).resolve())['sha256']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=review(a.inputs,a.repo)
    with a.out.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'passed':r['passed'],'verified':r['verified']}))


if __name__=='__main__':main()
