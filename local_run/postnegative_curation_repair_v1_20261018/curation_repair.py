#!/usr/bin/env python3
"""Exact label-free membership polynomial repair on natural E5 graphs.

Membership coordinates e_v are deterministic record identifiers, not task
labels. No learner or model head is constructed. Original IndexedState code
is imported unchanged; full states and request IDs remain inside this process.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('third_stage1_common',HERE/'stage1_common.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
need,load,save,sha=s.need,s.load,s.save,s.sha
CHECKPOINTS=(1,8,32,128)
HORIZON=128
MASTER_SEED=20261018
NAMESPACE='third-postnegative-curation-repair-v1'


def load_algorithm(repo):
    spec=importlib.util.spec_from_file_location('third_frozen_indexed',repo/'algorithm_round2.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def polynomial(graph,unit,unit_index,coordinate_index,dimension,horizon):
    """Independent assembly of each e_v (1-x_owner) product x_blocker.

    Uses exact integers and Boolean source idempotence. No invented labels.
    """
    coeff={}
    all_blockers=graph.blockers
    for i,rid in enumerate(graph.record_ids):
        owner=unit_index[graph.source_ids[i] if unit=='source' else rid]
        blockers={unit_index[graph.source_ids[int(j)] if unit=='source' else graph.record_ids[int(j)]] for j in all_blockers[i]}
        if owner in blockers: continue
        col=coordinate_index[rid]
        for key,sign in ((tuple(sorted(blockers)),1),(tuple(sorted(blockers|{owner})),-1)):
            if len(key)>horizon: continue
            if key not in coeff: coeff[key]=[0]*dimension
            coeff[key][col]+=sign
    return {key:tuple(values) for key,values in coeff.items() if any(values)}


def independent_selected(graph,unit,deleted_names):
    owners=graph.source_ids if unit=='source' else graph.record_ids
    blockers=graph.blockers
    return {rid for i,rid in enumerate(graph.record_ids) if owners[i] not in deleted_names and
            all(owners[int(j)] in deleted_names for j in blockers[i])}


def membership_ids(state,ids):
    value=state.constant();need(len(value)==len(ids) and all(v in (0,1) for v in value),'membership_not_binary')
    return {ids[i] for i,v in enumerate(value) if v}


def signature(state):
    return state.snapshot()


def one_arm(graph,x,arm,genuine,algorithm,build,structure):
    ids=list(graph.record_ids); sources=list(graph.source_ids)
    unit='source' if arm=='S' else 'record'
    universe=tuple(sorted(set(sources) if unit=='source' else ids)); ui={v:i for i,v in enumerate(universe)}
    ci={v:i for i,v in enumerate(ids)}
    seed=structure.derive_seed(MASTER_SEED,NAMESPACE,arm)
    trajectory=structure.generate_request_trajectory(graph,arm,seed,HORIZON,CHECKPOINTS,genuine_sources=genuine)
    horizon=trajectory.initial_horizon
    if not trajectory.deletion_order:
        return {'arm':arm,'status':'unavailable_empty_request_frame','requested_horizon':HORIZON,'effective_horizon':0,'checkpoints':[]}
    # All cutoffs reuse the same fixed frame and seed; R/S are graph-independent.
    order=list(trajectory.deletion_order);requests=[ui[u] for u in order]
    construction_start=time.monotonic()
    initial=polynomial(graph,unit,ui,ci,len(ids),horizon)
    def fresh_initial():return algorithm.IndexedState(len(universe),horizon,len(ids),range(len(universe)),initial)
    seq=fresh_initial();meter=seq.meter();initial_selected=membership_ids(seq,ids)
    need(initial_selected==independent_selected(graph,unit,set()),'initial_selection')
    construction_seconds=time.monotonic()-construction_start
    initial_keys=len(seq.coeff);initial_incidences=meter.initial_incidences
    deleted_names=set();rows=[];checkpoint_set=set(trajectory.checkpoints)
    singleton_checks=0;request_repair_seconds=0.0
    for step,(name,index) in enumerate(zip(order,requests),1):
        start=time.monotonic();seq.delete([index],meter);request_repair_seconds+=time.monotonic()-start
        deleted_names.add(name)
        expected=independent_selected(graph,unit,deleted_names)
        observed=membership_ids(seq,ids)
        need(observed==expected,'singleton_membership_mismatch')
        seq.check_invariants();algorithm.assert_bounds(seq,meter);singleton_checks+=1
        if step not in checkpoint_set:continue
        # Repeat the identical cumulative request as one batch and reverse order.
        batch=fresh_initial();batch_meter=batch.meter();batch.delete(requests[:step],batch_meter)
        reverse=fresh_initial();reverse_meter=reverse.meter()
        for u in reversed(requests[:step]):reverse.delete([u],reverse_meter)
        want=signature(seq)
        need(signature(batch)==want and signature(reverse)==want,'composition_snapshot_mismatch')
        batch.check_invariants();reverse.check_invariants()
        algorithm.assert_bounds(batch,batch_meter);algorithm.assert_bounds(reverse,reverse_meter)
        del batch,reverse
        # Fresh retained-data recuration recomputes the original scalar-order graph.
        keep=[i for i,rid in enumerate(ids) if (sources[i] if unit=='source' else rid) not in deleted_names]
        retained_ids=[ids[i] for i in keep];retained_sources=[sources[i] for i in keep]
        oracle_start=time.monotonic()
        fresh_graph=build(x[keep],retained_ids,graph.threshold,retained_sources,seed=0,block_size=37)
        oracle_selected={retained_ids[i] for i,b in enumerate(fresh_graph.blockers) if len(b)==0}
        need(oracle_selected==observed,'fresh_recuration_selection_mismatch')
        alive=set(range(len(universe)))-set(requests[:step]);remaining=horizon-step
        rebuilt=polynomial(fresh_graph,unit,ui,ci,len(ids),remaining)
        fresh_state=algorithm.IndexedState(len(universe),remaining,len(ids),alive,rebuilt)
        fresh_state.check_invariants()
        need(signature(fresh_state)==want,'fresh_retained_state_mismatch')
        need(membership_ids(fresh_state,ids)==oracle_selected,'fresh_membership_mismatch')
        # Frozen generic service uses idempotent retries. Test exactly that API.
        retry_before=signature(seq);seq.delete([requests[0]],meter)
        need(signature(seq)==retry_before,'idempotent_retry_mutation')
        rows.append({'checkpoint':step,'remaining_horizon':remaining,'deleted_records':len(ids)-len(keep),
            'selected_records':len(observed),'surviving_admissions':len(observed-initial_selected),
            'current_coefficient_keys':len(seq.coeff),'current_alive_units':len(seq.alive),
            'all_selected_membership_coordinates_equal':True,'all_coefficient_and_index_state_equal':True,
            'sequential_batch_reverse_equal':True,'fresh_retained_graph_equal':True,
            'retry_state_unchanged':True,'fresh_oracle_seconds':time.monotonic()-oracle_start})
        del fresh_graph,fresh_state,rebuilt,want
    return {'arm':arm,'unit':unit,'status':'completed','requested_horizon':HORIZON,'effective_horizon':horizon,
        'sampling_frame_size':trajectory.sampling_frame_size,'request_seed':seed,
        'request_order_digest':hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),
        'singleton_checks':singleton_checks,'initial_keys':initial_keys,'initial_incidences':initial_incidences,
        'construction_seconds':construction_seconds,'singleton_repair_seconds':request_repair_seconds,
        'meter':asdict(meter),'checkpoints':rows,
        'scope':'Exact integer membership statistics and canonical logical state; no learner, task labels, physical erasure or comparative speed claim.'}


def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();completed=[]
    try:
        need(sys.flags.optimize==0,'Python_assertions_required')
        protocol=load(args.protocol);need(sha(args.protocol)['sha256']==args.protocol_sha256,'protocol_hash')
        need(protocol['schema']=='third-postnegative-curation-repair-stage2-v1' and protocol['status']=='frozen_for_stage2','protocol_admission')
        need(sha(HERE/'stage1_common.py')==protocol['helper_binding'],'helper_binding')
        import numpy as np
        import scipy
        for key,value in (('python',platform.python_version()),('numpy',np.__version__),('scipy',scipy.__version__),('platform',platform.system())):
            need(value==protocol['runtime'][key],'runtime_'+key)
        def timed_out(*unused):raise TimeoutError('stage_wall_cap')
        signal.signal(signal.SIGALRM,timed_out);signal.alarm(protocol['resource_limits']['wall_seconds'])
        need(shutil.disk_usage(out).free>=protocol['resource_limits']['storage_reserve_bytes'],'scratch_reserve')
        repo=Path(args.repo).resolve();build,panels=s.imports(repo,protocol)
        algorithm=load_algorithm(repo)
        from ccu import structure
        paths={k:Path(getattr(args,k)).resolve() for k in ('allocation','prefixes','vectors')}
        bindings={k:sha(p) for k,p in paths.items()};need(bindings==protocol['inputs'],'original_input_bindings')
        rows=s.select_rows(load(paths['allocation']),load(paths['prefixes']),panels)
        cache=np.load(paths['vectors'],mmap_mode='r',allow_pickle=False)
        need(cache.shape==(140250,768) and cache.dtype==np.float32,'full_cache_shape')
        x=np.array(cache[[r['original_full_row_index'] for r in rows]],dtype=np.float32,order='C',copy=True)
        ids=[r['record_id'] for r in rows];sources=[r['source_unit_id'] for r in rows]
        genuine={r['source_unit_id'] for r in rows if r['genuine_source_request_eligible'] and r['source_kind'].startswith('native_')}
        save(out/'intake.json',{'input_bindings':bindings,'protocol_sha256':args.protocol_sha256,'rows':len(ids),
            'labels_read':False,'text_read':False,'heldout_feature_slices_read':False,
            'coordinate_definition':'e_v is the deterministic original-panel record membership coordinate, not an annotation or supervised target.',
            'runtime':protocol['runtime']})
        save(out/'subset_oracle.json',s.independent_small_oracle(x,ids,sources,build))
        request_digests={}
        for cutindex,cut in enumerate(s.GRID):
            graph=build(x,ids,float(cut),sources,seed=0,block_size=256)
            arms={arm:one_arm(graph,x,arm,genuine,algorithm,build,structure) for arm in ('R','S')}
            for arm,result in arms.items():
                if result['status']=='completed':
                    digest=result['request_order_digest'];need(arm not in request_digests or request_digests[arm]==digest,'threshold_changed_requests');request_digests[arm]=digest
            save(out/f'threshold_{cutindex:02d}.json',{'index':cutindex,'threshold_fraction':[cut.numerator,cut.denominator],
                'threshold':float(cut),'arms':arms,'all_executed_checks_passed':True,'semantic_validity_claim':False,'primary_study_result':False})
            completed.append(cutindex);del graph
            need(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=protocol['resource_limits']['output_cap_bytes'],'output_cap')
            need(shutil.disk_usage(out).free>=protocol['resource_limits']['storage_reserve_bytes'],'scratch_reserve')
        need({k:sha(p) for k,p in paths.items()}==bindings,'original_input_changed')
        for rel,binding in protocol['frozen_sources'].items():need(sha(repo/rel)==binding,'frozen_source_changed')
        save(out/'completion.json',{'status':'completed','threshold_indices':completed,'all51cuts_completed':completed==list(range(51)),
             'model_or_human_annotation_calls':0,'task_learner_trained':False,'v4_failure_retained':True,
             'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             'kernel_no_reaccess_isolation_claim':False,'supervised_prediction_effect_claim':False,
             'outputs':{p.name:sha(p) for p in sorted(out.glob('*.json'))}})
        print(json.dumps({'status':'completed','thresholds':len(completed)}))
    except BaseException as error:
        save(out/'failure.json',{'status':'failed','exception_type':type(error).__name__,'reason':str(error),
            'completed_threshold_indices':completed,'partial_results_not_complete':True,'automatic_retry':False,'elapsed_seconds':time.monotonic()-start})
        raise
    finally:signal.alarm(0)


def main():
    ap=argparse.ArgumentParser()
    for name in ('repo','protocol','protocol-sha256','allocation','prefixes','vectors','out'):ap.add_argument('--'+name,required=True)
    run(ap.parse_args())

if __name__=='__main__':main()
