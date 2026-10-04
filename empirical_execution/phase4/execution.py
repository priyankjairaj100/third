#!/usr/bin/env python3
"""Fair fixed-input ridge integration. Timings are plumbing diagnostics, not speed evidence.

Primary execution is deliberately unavailable in this harness until the complete
protocol implementation, isolation and externally reviewed intake are integrated.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import dataclasses,hashlib,json,math,platform,time
import numpy as np
from ccu.core import EligiblePayloadState,RidgeMoments,ridge_moments,solve_ridge
from ccu.compact_payload import CompactEligiblePayloadState
from ccu.joint_span_summary import JointSpanRidgeSummary
from ccu.data import sha256_file
from phase3.reference_graph import build_reference_graph,id_edges

METHODS={
 'O-G':'Fresh retained reference graph + fresh selected moments + common solve',
 'O-T':'Correct oracle selected IDs + fresh moments + common solve; excludes graph work',
 'B-E-signed-python':'Existing string-indexed eligible payload with signed batched moment updates',
 'B-E-compact-common':'Integer-indexed eligible payload; rebuild selected moments at release',
 'P-I-jointspan-common':'Indexed joint-span aggregate summary; recover moments at release',
 'B-F':'Fresh ridge on retained initially selected records; intentionally wrong curation target'}
UNSUPPORTED={
 'S':'This runner presently supports record-service paths only; source requests are not relabeled',
 'B-A':'All-current-payload general service not integrated',
 'P-S':'Full coefficient-map scan comparator not integrated',
 'P-R':'Protocol rank-coordinate scan comparator not integrated; joint-span P-I is distinct',
 'O-R':'Standard SemDeDup complete refit branch not integrated',
 'B-E-optimized-CSR':'Current signed-payload branch is Python dictionary/set code, not the protocol optimized CSR performance baseline',
 'compact-native-decoders':'Optional dual/joint-span decoders excluded here to hold decoder identical',
 'primary-study':'Requires full registered branches, real assets/ratings, independent workers and externally reviewed execution lock'}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(value):return hashlib.sha256(value).hexdigest()
def array_hash(value):
    a=np.ascontiguousarray(value)
    return sha(str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def append(path,value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
def elapsed(call):
    started=time.perf_counter();value=call();return value,time.perf_counter()-started


def independent_edges(features,record_ids,threshold,deleted=(),seed=0):
    """Scalar FP64 oracle independently reproducing the specified coordinate order.

    Uses neither production normalization, score, graph, nor selected-ID routines.
    Numeric sqrt is NumPy scalar sqrt so platform target matches the reference.
    """
    ids=list(record_ids);dead=set(deleted)
    order=sorted(range(len(ids)),key=lambda i:(hashlib.sha256(f'priority-v1|{seed}|{ids[i]}'.encode()).digest(),ids[i]))
    order=[i for i in order if ids[i] not in dead];norm={}
    for i in order:
        row=[np.float64(v) for v in features[i]];s=np.float64(0)
        for v in row:s=np.float64(s+np.float64(v*v))
        den=np.sqrt(s);norm[i]=[np.float64(v/den) for v in row]
    edges=set();selected=[]
    for pos,i in enumerate(order):
        blocked=False
        for j in order[:pos]:
            score=np.float64(0)
            for a,b in zip(norm[i],norm[j]):score=np.float64(score+np.float64(a*b))
            if score>threshold:edges.add((ids[i],ids[j]));blocked=True
        if not blocked:selected.append(i)
    return edges,selected


def common_decode(moments,lambda_reg):
    # At zero count the specified target is zero; expose any cancellation residue
    # in discrepancy logs, then use the same explicit empty-objective convention.
    if moments.count==0:
        moments=RidgeMoments(np.zeros_like(moments.gram),np.zeros_like(moments.cross),0)
    return solve_ridge(moments,lambda_reg)


def discrepancy(got,wanted):
    return {'count_equal':got.count==wanted.count,
      'gram_max_abs':float(np.max(np.abs(got.gram-wanted.gram),initial=0)),
      'cross_max_abs':float(np.max(np.abs(got.cross-wanted.cross),initial=0)),
      'gram_relative_fro':float(np.linalg.norm(got.gram-wanted.gram)/max(1,np.linalg.norm(wanted.gram))),
      'cross_relative_fro':float(np.linalg.norm(got.cross-wanted.cross)/max(1,np.linalg.norm(wanted.cross)))}


def input_bindings(cx,x,y,ids,request_manifest,threshold,lambda_reg):
    return {'curator_array_sha256':array_hash(cx),'learner_array_sha256':array_hash(x),
      'target_array_sha256':array_hash(y),'record_ids_sha256':sha(canonical(list(ids))),
      'request_manifest_sha256':request_manifest['manifest_sha256'],
      'threshold_hex':float(threshold).hex(),'lambda_reg_hex':float(lambda_reg).hex(),
      'execution_code_sha256':sha256_file(Path(__file__))}


def run_comparison(curator_x,learner_x,targets,record_ids,request_manifest,output_dir,*,
                   threshold,lambda_reg,source_ids=None,evidence_role='engineering_nonconfirmatory',
                   primary_lock=None,block_size=32,tolerance=1e-9):
    """Execute every locked record-service path; preserve failed and unsupported cells.

    Input arrays remain evaluator-owned for independent audits: this process is
    not evidence of service no-reaccess, peak memory, or cold-cache performance.
    """
    from phase4.requests import verify_manifest
    if evidence_role!='engineering_nonconfirmatory':
        raise RuntimeError('Primary execution blocked: this integration harness is engineering-only; a metadata declaration or lock cannot waive unfinished protocol branches/isolation')
    if primary_lock is not None:raise ValueError('Primary lock cannot be used to promote engineering output')
    cx=np.asarray(curator_x);x=np.asarray(learner_x);y=np.asarray(targets)
    ids=tuple(record_ids);n=len(ids)
    if y.ndim==1:y=y[:,None]
    if cx.dtype!=np.float32 or x.dtype!=np.float32 or y.dtype!=np.float64:
        raise ValueError('Exact supplied FP32 curator/learner and FP64 target arrays required')
    if cx.ndim!=2 or x.ndim!=2 or y.ndim!=2 or len(cx)!=n or len(x)!=n or len(y)!=n or min(cx.shape[1],x.shape[1],y.shape[1])<1:
        raise ValueError('Aligned nonzero dimensions required')
    if not all(np.isfinite(a).all() for a in (cx,x,y)):raise ValueError('Nonfinite input')
    if len(set(ids))!=n or any(not isinstance(i,str) or not i for i in ids):raise ValueError('Unique IDs required')
    if isinstance(lambda_reg,bool) or not math.isfinite(lambda_reg) or lambda_reg<=0:raise ValueError('Positive lambda required')
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Finite positive nonboolean audit tolerance required')
    graph,graph_build_seconds=elapsed(lambda:build_reference_graph(cx,ids,threshold,source_ids,block_size=block_size))
    verify_manifest(request_manifest,graph)
    if request_manifest.get('evidence_role',evidence_role)!=evidence_role:raise ValueError('Request scope mismatch')
    trajectories=request_manifest['trajectories'];names=[r['trajectory_id'] for r in trajectories]
    if len(names)!=len(set(names)):raise ValueError('Duplicate trajectory IDs')
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite frozen execution directory')
    out.mkdir(parents=True,exist_ok=True)
    # This lock is written before method construction, repairs, or head outcomes.
    bindings=input_bindings(cx,x,y,ids,request_manifest,threshold,lambda_reg)
    lock={'schema':'ccu-phase4-engineering-execution-1','evidence_role':evidence_role,
      'confirmatory_study_ready':False,'bindings':bindings,'methods':METHODS,'unsupported':UNSUPPORTED,
      'lambda_regularizer':'lambda * current selected count; all coordinates penalized; no intercept',
      'common_decoder':'core.solve_ridge FP64 primal Cholesky; empty selection returns zero',
      'moment_comparison_tolerance':tolerance,'checkpoint_failures':'preserve row and mark later path checkpoints not executed; never reroll',
      'timing_scope':'single process wall time; shared inputs and warm caches; audits excluded; no benchmark claim',
      'memory_scope':'state numeric arrays and recursive owned-byte estimates; no RSS/peak/allocator claim',
      'input_storage_shared_evaluator_bytes':int(cx.nbytes+x.nbytes+y.nbytes),
      'python':platform.python_version(),'numpy':np.__version__,
      'module_sha256':{name:sha256_file(ROOT/name) for name in ['phase4/execution.py','phase4/requests.py','phase3/reference_graph.py','phase3/panels.py','ccu/core.py','ccu/summary.py','ccu/compact_payload.py','ccu/joint_span_summary.py']}}
    write(out/'execution_lock.json',lock);write(out/'requests.json',request_manifest)
    for name in ['construction.jsonl','checkpoints.jsonl','heads.jsonl','audits.jsonl']: (out/name).write_text('')
    oracle_edges,oracle_initial=independent_edges(cx,ids,threshold)
    if id_edges(graph)!=oracle_edges:raise AssertionError('Initial graph disagrees with independent scalar oracle')
    initial=set(oracle_initial);lookup={name:i for i,name in enumerate(ids)}
    audit_initial={'initial_graph_edges':len(oracle_edges),'initial_selected_count':len(initial),
      'independent_scalar_graph_equal':True,'phase3_graph_construction_seconds':graph_build_seconds,
      'selected_ids':[ids[i] for i in sorted(initial)]}
    write(out/'initial_graph_audit.json',audit_initial)
    total_rows=0;failed=0;zero_admission=0;max_head=0.;max_moment=0.;unsupported=0
    for path in trajectories:
        pid=path['trajectory_id'];arm=path['arm'];position=0;path_failed=False
        if path['unit']!='record':
            for checkpoint in path['checkpoints']:
                append(out/'checkpoints.jsonl',{'trajectory_id':pid,'arm':arm,'checkpoint':checkpoint,'status':'unsupported_source_service','reason':UNSUPPORTED['S']});unsupported+=1
            continue
        order=path['deletion_order'];h=path['initial_horizon']
        if len(set(order))!=len(order) or not set(order)<=set(ids) or len(order)>h or not 0<=h<=n:
            raise ValueError('Invalid locked trajectory')
        if path['checkpoints']!=sorted(set(path['checkpoints'])) or any(not 0<k<=len(order) for k in path['checkpoints']):raise ValueError('Invalid checkpoints')
        states={}
        for method,constructor in [('B-E-signed-python',lambda:EligiblePayloadState(graph,x,y,h)),
          ('B-E-compact-common',lambda:CompactEligiblePayloadState.build(x,y,graph.blockers,h)),
          ('P-I-jointspan-common',lambda:JointSpanRidgeSummary.build(x,y,graph.blockers,h))]:
            try:
                state,seconds=elapsed(constructor);states[method]=state
                state.check_invariants()
                append(out/'construction.jsonl',{'trajectory_id':pid,'method':method,'status':'completed','seconds':seconds,'state':state.accounting()})
            except Exception as error:
                append(out/'construction.jsonl',{'trajectory_id':pid,'method':method,'status':'failed','error_type':type(error).__name__,'error':str(error)})
                path_failed=True;failed+=1;break
        for checkpoint in path['checkpoints']:
            context={'trajectory_id':pid,'arm':arm,'checkpoint':checkpoint,'dimension':x.shape[1]}
            total_rows+=1
            if path_failed:
                append(out/'checkpoints.jsonl',{**context,'status':'not_executed_after_path_failure'});continue
            methods={}
            try:
                deleted=order[:checkpoint];batch=order[position:checkpoint];dead={lookup[i] for i in deleted}
                ix=np.asarray([i for i in range(n) if i not in dead],dtype=int)
                fresh,graph_seconds=elapsed(lambda:build_reference_graph(cx[ix],[ids[i] for i in ix],threshold,
                    None if source_ids is None else [source_ids[i] for i in ix],block_size=block_size))
                independent,selected=independent_edges(cx,ids,threshold,deleted)
                if id_edges(fresh)!=independent:raise AssertionError('Fresh retained graph disagrees with scalar oracle')
                selected=set(selected);local_selected={int(ix[j]) for j in fresh.selected_indices()}
                if local_selected!=selected:raise AssertionError('Fresh graph selection disagrees with scalar oracle')
                sorted_selected=np.asarray(sorted(selected),dtype=int)
                oracle_moments,moment_seconds=elapsed(lambda:ridge_moments(x[sorted_selected],y[sorted_selected]))
                oracle,decode_seconds=elapsed(lambda:common_decode(oracle_moments,lambda_reg))
                added=selected-initial;zero_admission+=not added
                methods={'O-G':{'graph_seconds':graph_seconds,'moments_seconds':moment_seconds,'decode_seconds':decode_seconds,
                    'total_seconds':graph_seconds+moment_seconds+decode_seconds,'count':oracle_moments.count},
                    'O-T':{'moments_seconds':moment_seconds,'decode_seconds':decode_seconds,'total_seconds':moment_seconds+decode_seconds,
                    'count':oracle_moments.count,'shared_with_OG':'Exactly same fresh-moment/solve execution, split accounting, not an independent timing replicate'}}
                head_records={'O-G':oracle.weights.tolist(),'O-T':oracle.weights.tolist()}
                for method,state in states.items():
                    identifiers=batch if method=='B-E-signed-python' else [lookup[i] for i in batch]
                    meter,repair_seconds=elapsed(lambda:state.delete(identifiers))
                    moments,recover_seconds=elapsed(state.moments)
                    head,solve_seconds=elapsed(lambda:common_decode(moments,lambda_reg))
                    state.check_invariants()
                    discrepancy_row=discrepancy(moments,oracle_moments)
                    diff=float(np.max(np.abs(head.weights-oracle.weights),initial=0))
                    max_head=max(max_head,diff);max_moment=max(max_moment,discrepancy_row['gram_max_abs'],discrepancy_row['cross_max_abs'])
                    if not discrepancy_row['count_equal'] or max(discrepancy_row['gram_relative_fro'],discrepancy_row['cross_relative_fro'],diff)>tolerance:
                        raise AssertionError(f'{method} exceeds numerical audit tolerance: {discrepancy_row}, head={diff}')
                    if method=='B-E-signed-python':membership={lookup[i] for i in state.selected_ids()}
                    elif method=='B-E-compact-common':membership=set(state.selected_ids())
                    else:membership=None
                    if membership is not None and membership!=selected:raise AssertionError('Eligible payload selected membership mismatch')
                    if state.horizon!=h-checkpoint:raise AssertionError('Incorrect remaining horizon')
                    alive=set(state.alive_units) if method=='B-E-signed-python' else {ids[i] for i in state.alive}
                    if alive!=set(ids)-set(deleted):raise AssertionError('Incorrect live service IDs')
                    if dataclasses.is_dataclass(meter):meter=dataclasses.asdict(meter)
                    methods[method]={'repair_seconds':repair_seconds,'moments_recovery_seconds':recover_seconds,
                      'decode_seconds':solve_seconds,'total_seconds':repair_seconds+recover_seconds+solve_seconds,
                      'state':state.accounting(),'meter':meter,'discrepancy':discrepancy_row,'max_head_abs_difference':diff,
                      'selected_membership_equal':True if membership is not None else 'statistic_only_no_ID_decoder',
                      'state_invariants_passed':True}
                    head_records[method]=head.weights.tolist()
                frozen_ix=np.asarray(sorted(initial-dead),dtype=int)
                fm,fs=elapsed(lambda:ridge_moments(x[frozen_ix],y[frozen_ix]))
                fw,fd=elapsed(lambda:common_decode(fm,lambda_reg))
                frozen_diff=float(np.max(np.abs(fw.weights-oracle.weights),initial=0))
                if not added and frozen_diff>tolerance:raise AssertionError('Zero-admission control differs')
                methods['B-F']={'moments_seconds':fs,'decode_seconds':fd,'total_seconds':fs+fd,'count':fm.count,
                    'wrong_target_diagnostic':True,'head_max_abs_from_correct_target':frozen_diff}
                head_records['B-F']=fw.weights.tolist()
                row={**context,'status':'completed_engineering','deleted_record_ids':deleted,
                  'selected_record_ids':[ids[i] for i in sorted(selected)],'added_record_ids':[ids[i] for i in sorted(added)],
                  'removed_initial_selected_ids':[ids[i] for i in sorted(initial&dead)],'addition_count':len(added),
                  'remaining_horizon':h-checkpoint,'methods':methods}
                append(out/'checkpoints.jsonl',row);append(out/'heads.jsonl',{**context,'weights':head_records})
                append(out/'audits.jsonl',{**context,'independent_scalar_retained_edges_equal':True,
                  'retained_edges':len(independent),'selected_count':len(selected),'same_fp32_arrays_targets_lambda_and_decoder':True,
                  'full_coefficient_canonical_bytes_checked':False,'roundoff_certificate_executed':False,
                  'selected_moments_and_head_checked':True,'audit_work_excluded_from_method_timings':True})
                position=checkpoint
            except Exception as error:
                failed+=1;path_failed=True
                append(out/'checkpoints.jsonl',{**context,'status':'failed','error_type':type(error).__name__,'error':str(error),
                  'completed_method_records_before_failure':methods})
    summary={'status':'completed_engineering' if not failed else 'engineering_with_failures','checkpoint_rows':total_rows,
      'failure_events':failed,'unsupported_checkpoint_rows':unsupported,'zero_admission_rows':int(zero_admission),
      'maximum_same_target_head_abs_difference':max_head,'maximum_same_target_moment_abs_difference':max_moment,
      'primary_study_started':False,'semantic_encoder_identity_verified_by_this_runner':False,
      'semantic_or_utility_effect_claim':False,'speedup_or_lifecycle_claim':False,'process_peak_memory_measured':False,
      'evaluator_and_services_process_isolated':False,'roundoff_certificate_claim':False,
      'execution_lock_sha256':sha256_file(out/'execution_lock.json'),'bindings':bindings}
    write(out/'summary.json',summary);return summary
