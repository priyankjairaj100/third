#!/usr/bin/env python3
"""Independent checks for Phase 6 contracts and preserved evidence.

Small mathematical fixtures check software only. They are not empirical data.
No test can establish missing corpus provenance or genuine human collection.
"""
from __future__ import annotations
import argparse
import copy
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'empirical_execution'))
CHECKS = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def rejects(name, function, exceptions=(ValueError, RuntimeError, PermissionError)):
    try:
        function()
    except exceptions:
        check(name, True)
        return
    raise AssertionError(name + ': unexpectedly accepted')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def independent_hypergeometric(N, K, n, x):
    """Exact design probability, independent of SciPy or production helpers."""
    if not max(0, n - (N-K)) <= x <= min(n, K):
        return Fraction(0)
    return Fraction(math.comb(K, x) * math.comb(N-K, n-x), math.comb(N, n))


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / 'empirical_execution/phase6').glob('*.py'))}


def human_interval_checks():
    from phase6 import human_inference as hi
    comparisons = 0
    coverage_cases = 0
    least_coverage = Fraction(1)
    alpha = Fraction(1, 10)
    # Enumerate the exact randomization law. No production probability is used.
    for N in range(1, 13):
        for n in range(N+1):
            for K in range(N+1):
                coverage = Fraction(0)
                for x in range(n+1):
                    probability = independent_hypergeometric(N, K, n, x)
                    for upper in (False, True):
                        expected = sum((independent_hypergeometric(N, K, n, j)
                                        for j in range(n+1)
                                        if (j >= x if upper else j <= x)), Fraction(0))
                        if hi.hypergeom_tail(N,K,n,x,upper=upper) != expected:
                            raise AssertionError(('exact hypergeometric tail',N,K,n,x,upper))
                        comparisons += 1
                    interval = hi.count_interval(N,n,x,0,alpha=alpha)
                    coverage += probability * (interval[0] <= K <= interval[1])
                    # Every possible outcome-dependent erasure must widen the set.
                    for missing_positive in range(x+1):
                        for missing_negative in range(n-x+1):
                            widened = hi.count_interval(N,n,x-missing_positive,
                                       missing_positive+missing_negative,alpha=alpha)
                            if not (widened[0] <= interval[0] <= interval[1] <= widened[1]):
                                raise AssertionError('arbitrary missingness failed set containment')
                if coverage < 1-alpha:
                    raise AssertionError(('exact marginal coverage',N,K,n,coverage))
                least_coverage = min(least_coverage,coverage)
                coverage_cases += 1
    check('human exact hypergeometric tails independently enumerated', comparisons > 0)
    check('human finite-population coverage independently enumerated', coverage_cases > 0)
    check('human arbitrary missingness preserves every complete-data interval', True)
    check('human no sample gives complete unidentified count range',
          hi.count_interval(17,0,0,0)==[0,17])
    check('human census gives exact count', hi.count_interval(17,17,9,0)==[9,9])
    # Overlapping SRS routes are not an SRS union. Check the owner construction.
    population=set('abcd')
    frames=[set('abc'),set('bcd')]
    owners={'a':0,'b':0,'c':0,'d':1}
    overlap_cases=0
    for bits in itertools.product((0,1), repeat=4):
        outcomes=dict(zip(sorted(population),bits));K=sum(bits)
        for hide_positives in (False,True):
            coverage=Fraction(0)
            for first in itertools.combinations(sorted(frames[0]),2):
                for second in itertools.combinations(sorted(frames[1]),2):
                    sample=set(first)|set(second)
                    values={i:(None if hide_positives and outcomes[i] else outcomes[i])
                            for i in sample}
                    strata=[{'index':h,'frame':frames[h],'sample':set(s),'N':3,'n':2}
                            for h,s in enumerate((first,second))]
                    result=hi.category_interval(population,values,strata,owners,alpha=alpha)
                    interval=result['count_interval']
                    coverage += Fraction(int(interval is not None and interval[0]<=K<=interval[1]),9)
            if coverage<1-alpha:
                raise AssertionError(('overlap owner coverage',bits,hide_positives,coverage))
            overlap_cases+=1
    check('human overlapping-route owner construction has exact design coverage',overlap_cases==32)
    rejects('human fractional count rejected',lambda:hi.count_interval(10,3,1.2,0))
    rejects('human impossible missing count rejected',lambda:hi.count_interval(10,3,2,2))
    rejects('human zero alpha rejected',lambda:hi.count_interval(10,3,2,0,alpha=0))
    return {'evidence_role':'mathematical software checks',
            'independent_tail_comparisons':comparisons,
            'independent_coverage_cases':coverage_cases,
            'least_observed_exact_coverage':str(least_coverage),
            'overlapping_route_coverage_cases':overlap_cases,
            'actual_human_responses_created':0}


def method_checks(temporary):
    from ccu.core import BlockerGraph
    from phase6.methods import FrozenSelection, FP32Payload
    from phase5.decoders import decode, DEFAULT_POLICY
    from phase6.measurement import state_inventory,serialized_inventory,logical_update_meter,SnapshotReadMeter
    blockers=[[],[0],[0],[1,2],[]]
    ptr=[0];flat=[]
    for row in blockers:
        flat.extend(row);ptr.append(len(flat))
    ids=tuple('r'+str(i) for i in range(5))
    sources=('s0','s0','s1','s2','s2')
    graph=BlockerGraph(ids,sources,np.arange(5,dtype=np.int64),
                       np.asarray(ptr,np.int64),np.asarray(flat,np.int64),.6)
    x=np.asarray([[1,.5],[.25,-.5],[-.25,.75],[.5,-1],[.75,.25]],np.float32)
    y=np.asarray([[.25,1],[.5,0],[.75,1],[1,0],[0,1]],np.float64)
    checks=0
    maximum_fp32_moment_error=0.
    for unit,owners in [('record',ids),('source',sources)]:
        universe=sorted(set(owners))
        for size in range(min(3,len(universe))+1):
            for deleted in itertools.combinations(universe,size):
                state=FrozenSelection(graph,x,y,3,unit=unit)
                state.delete(deleted)
                expected=[i for i,B in enumerate(blockers) if not B and owners[i] not in deleted]
                moments=state.moments()
                if moments.count!=len(expected) or not np.array_equal(moments.gram,x[expected].T.astype(float)@x[expected].astype(float)):
                    raise AssertionError(('B-F wrong-target moments',unit,deleted))
                if not np.array_equal(moments.cross,x[expected].T.astype(float)@y[expected]):
                    raise AssertionError('B-F cross moments')
                snapshot=temporary/(unit+str(checks)+'.npz');state.snapshot(snapshot)
                restored=FrozenSelection.load(snapshot)
                if not np.array_equal(restored.moments().cross,moments.cross):
                    raise AssertionError('B-F restored target')
                fp32=FP32Payload(graph,x,y,3,unit=unit)
                operation=fp32.delete(deleted)
                selected=[i for i,B in enumerate(blockers)
                          if owners[i] not in deleted and all(owners[j] in deleted for j in B)]
                fm=fp32.moments()
                gram=x[selected].T.astype(float)@x[selected].astype(float)
                cross=x[selected].T.astype(float)@y[selected]
                error=max(np.max(np.abs(gram-fm.gram)),np.max(np.abs(cross-fm.cross)))
                maximum_fp32_moment_error=max(maximum_fp32_moment_error,float(error))
                if fm.count!=len(selected) or error!=0:
                    # These are exactly representable dyadic fixtures, not a relaxed numerical gate.
                    raise AssertionError(('FP32 dyadic exactness fixture',unit,deleted,error))
                snapshot=temporary/('fp32'+unit+str(checks)+'.npz');fp32.snapshot(snapshot)
                restored=FP32Payload.load(snapshot)
                if restored.gram.dtype!=np.float32 or not np.array_equal(restored.gram,fp32.gram):
                    raise AssertionError('FP32 snapshot precision changed')
                solver=decode(fm,.01)
                if not solver.diagnostics['release_allowed']:
                    raise AssertionError('FP32 stored-moment decoder unexpectedly failed dyadic fixture')
                meter=logical_update_meter(fp32,operation)
                if meter['hardware_memory_traffic_bytes'] is not None or meter['metadata_read_traffic_bytes'] is not None:
                    raise AssertionError('unmeasured physical or metadata traffic filled')
                checks+=1
    check('B-F target and FP32 own precision checked across record/source subsets',checks>0)
    check('B-F and FP32 snapshots preserve their own target and precision',True)
    check('FP32 uses the unchanged common residual gate',DEFAULT_POLICY['normalized_residual_limit']==1e-10)
    state=FrozenSelection(graph,x,y,3)
    before=state.snapshot_bytes()
    rejects('B-F malformed mixed request is atomic',lambda:state.delete(['r0','missing']))
    check('B-F malformed request leaves identical snapshot',before==state.snapshot_bytes())
    state.delete(['r1']);count=state.horizon
    state.delete(['r1','r1'])
    check('B-F repeated request consumes no new budget',count==state.horizon)
    root=np.arange(24,dtype=np.float32)
    inventory=state_inventory({'x':root,'view':root[::2],'alias':root})
    check('state inventory counts array roots once',inventory['unique_ndarray_root_buffer_bytes']==root.nbytes)
    check('state inventory keeps allocator and physical traffic unknown',inventory['python_allocator_bytes'] is None and inventory['hardware_memory_traffic_bytes'] is None)
    raw=state.snapshot_bytes();serialized=serialized_inventory({'state.npz':raw})
    row=serialized['files'][0]
    check('serialization member bytes plus headers reconcile',row['container_overhead_bytes']+sum(m['compressed_bytes'] for m in row['members'])==len(raw))
    binary=temporary/'meter_state.npz';binary.write_bytes(raw)
    meter=SnapshotReadMeter([binary])
    with meter:
        with binary.open('rb') as stream:
            one=stream.read();stream.seek(0);two=stream.read()
    measured=meter.report()
    check('snapshot read meter counts repeated actual binary reads',
          measured['returned_binary_read_bytes_total']==2*len(raw) and one==two==raw)
    check('snapshot read categories form a disjoint byte partition',
          sum(measured['returned_binary_read_bytes_by_category'].values())==2*len(raw) and
          measured['returned_binary_read_bytes_by_category']['payload_array_bytes']==2*(state.x.nbytes+state.y.nbytes))
    check('snapshot read meter states its observed API boundary',
          measured['C_level_bypasses_not_observed'] and not measured['kernel_storage_or_memory_traffic_claim'])
    return {'evidence_role':'algebraic software checks', 'states':checks,
            'maximum_FP32_dyadic_moment_error':maximum_fp32_moment_error,
            'residual_tolerance_unchanged':1e-10,
            'physical_memory_traffic_measured':False}


def recipe_checks():
    from phase6 import recipes
    from phase5 import study_registry as previous
    book=recipes.build_recipe_book();registry,jobs=recipes.build_registry()
    legacy_registry,legacy_jobs=previous.build_registry()
    original={j['job_id']:j for j in legacy_jobs if j.get('kind')!='unresolved_protocol_obligation'}
    current={j['job_id']:j for j in jobs}
    check('recipes retain every original A-G job without change',all(current.get(k)==v for k,v in original.items()))
    check('recipes replace exactly the 21 explicit prior obligations',
          len(book['recipes'])==21 and {r['id'] for r in book['recipes']}=={r[0] for r in previous.UNRESOLVED})
    check('recipes never create empirical evidence or activate inputs',
          all(not r['scientific_evidence_created'] for r in book['recipes']) and not registry['execution_allowed'])
    check('recipes retain unique job identities',len(current)==len(jobs))
    check('recipes bind compiled job content',registry['jobs_content_sha256']==recipes.digest(jobs))
    policies={r['id']:r['policy'] for r in book['recipes']}
    check('recipes keep unchanged numerical release tolerance',policies['primary_numerical_release_gate']['eta_maximum']==1e-10)
    check('recipes keep zero-start CG and finite iteration policy',policies['boundary_method_scope']['CG']==
          {'initialization':'zero','maximum_iterations':'10*dimension','eta':1e-10,'preconditioner':None,'warm_start':False})
    check('recipes keep human quality and independent validation requirements',
          policies['human_threshold_calibration']['judgments_each']==3 and
          policies['human_threshold_calibration']['validation_max_pairs_each']==200 and
          policies['human_threshold_calibration']['selection_max_pairs_each']==600)
    check('recipes separate optional deactivation from required results',
          policies['official_Civil_splits_optional']['activated'] is False)
    check('recipes retain valid-test-only Holm inputs',policies['test_source_bootstrap']['directional_p_values']==
          'only_valid_test_outputs_never_derive_from_percentile_CI')
    for N in (0,1,99,100,101,12799,12800,200000):
        value=recipes.one_percent_horizon(N)
        expected=(N+99)//100
        check('one-percent horizon includes true endpoint '+str(N),value['initial_horizon']==expected and
              value['checkpoints']==sorted({min(k,expected) for k in (1,8,32,128,expected) if min(k,expected)>0}))
    rejects('one-percent negative population refused',lambda:recipes.one_percent_horizon(-1))
    rejects('PPS noninteger mass refused',lambda:recipes.pps_order_exact_integer_tickets({'a':1.5},draw_count=1,seed_value=3))
    check('PPS uses distinct native source identities',len(set(recipes.pps_order_exact_integer_tickets({'a':1,'b':2,'c':3},draw_count=8,seed_value=4)))==3)
    rejects('development and confirmation cannot overlap',lambda:recipes.precision_analysis(
        [0.1]*16,development_ids=[str(i) for i in range(16)],confirmation_ids=['1']))
    return {'original_A_G_jobs_retained':len(original),'compiled_jobs':len(jobs),
            'added_recipe_jobs':len(jobs)-len(original),'all_21_policies_present':True,
            'recipe_compilation_is_not_scientific_execution':True}


def logistic_checks():
    from empirical_execution.phase4 import model_selection as ms
    from empirical_execution.phase4.check_model_selection import fixtures,provenance
    from phase6 import logistic_selection as ls
    from scipy.special import expit
    rows=fixtures('askubuntu');prov=provenance('askubuntu')
    x=np.asarray([[i/10,(i%3)-1,1] for i in range(10)],np.float32)
    vocabulary=ms.select_tag_vocabulary(rows,prov)
    ridge=ms.select_ridge_regularization(x,rows,prov,vocabulary_lock=vocabulary)
    selected=ls.select_logistic_decision_thresholds(x,rows,prov,ridge,vocabulary)
    lookup={r['record_id']:i for i,r in enumerate(rows)}
    ordered=x[[lookup[rid] for rid in selected['record_ids']]].astype(float)
    targets=np.asarray(ridge['calibration_targets']);folds=np.asarray(selected['record_folds'])
    actual=np.asarray(selected['selected_oof_probabilities']);expected=np.empty_like(actual)
    # A separate plain Newton implementation checks the fitted probability target.
    for fold in range(5):
        tr=folds!=fold;va=~tr;z=ordered[tr];y=targets[tr]
        for c in range(y.shape[1]):
            w=np.zeros(z.shape[1]);lam=selected['lambda']
            for iteration in range(100):
                p=expit(z@w);g=z.T@(p-y[:,c])/len(z)+lam*w
                H=z.T@(z*(p*(1-p))[:,None])/len(z)+lam*np.eye(z.shape[1])
                if np.linalg.norm(g)<1e-12:break
                w-=np.linalg.solve(H,g)
            if np.linalg.norm(g)>=1e-12:raise AssertionError('Independent calibration Newton failed')
            expected[va,c]=expit(ordered[va]@w)
    discrepancy=float(np.max(np.abs(expected-actual)))
    check('logistic probabilities match independent per-output Newton target',discrepancy<1e-8)
    check('logistic calibration keeps ridge lambda and genuine source folds',
          selected['lambda']==ridge['lambda'] and selected['record_folds']==ridge['record_folds'])
    for c,rule in enumerate(selected['thresholds']):
        positives=int(targets[:,c].sum())
        if positives==0:
            if rule['kind']!='never_positive':raise AssertionError('zero-positive logistic output')
            continue
        ranking=[]
        for threshold in sorted(set(actual[:,c])):
            prediction=actual[:,c]>=threshold
            f1=Fraction(2*int(np.sum(prediction*targets[:,c])),positives+int(prediction.sum()))
            ranking.append((f1,threshold))
        if rule['threshold']!=max(ranking)[1]:raise AssertionError('logistic exact F1 threshold or strict tie')
    check('logistic threshold objective and strict tie independently reconstructed',True)
    refuses=ms.select_tag_decision_thresholds(ridge)
    rejects('logistic evaluator refuses ridge decision lock',lambda:ls.apply_logistic_thresholds(actual,refuses))
    check('logistic selection never claims human or source authenticity',
          not selected['external_provenance_verified_by_this_module'] and not selected['confirmatory_study_ready'])
    return {'scope':'algebraic source/tag software fixture; no Stack empirical data',
            'folds':5,'outputs':20,'independent_probability_maximum_error':discrepancy}


def dyadic_certificate_checks():
    from phase4 import convex as original
    from phase6 import dyadic_convex as fast
    from phase5 import convex_multioutput as multi
    semantic=('gradient_lower','gradient_upper','gradient_norm_squared_upper',
              'parameter_error_squared_upper','parameter_error_norm_upper',
              'objective_gap_upper','meets_parameter_tolerance','work')
    cases=[(np.asarray([[.1,-.2,0],[.3,.4,-.5]],np.float32),np.asarray([.125,.875]),np.asarray([.17,-.33,.7])),
           (np.empty((0,3),np.float32),np.empty(0),np.asarray([.17,-.33,.7])),
           (np.asarray([[1,-1,0]],np.float32),np.asarray([np.nextafter(0.,1.)]),np.asarray([0.,0.,0.])),
           (np.asarray([[np.nextafter(np.float32(0),np.float32(1)),1,-1]],np.float32),
            np.asarray([1.]),np.asarray([np.nextafter(0.,1.),-.1,.1]))]
    compared=0
    for x,y,w in cases:
        for bits in (16,64,128):
            reference=original.certify_logistic(x,y,.03,w,bits=bits)
            result=fast.certify_logistic(x,y,.03,w,bits=bits)
            if any(reference[k]!=result[k] for k in semantic):
                raise AssertionError(('dyadic exact endpoint identity',len(x),bits))
            compared+=1
    check('dyadic certificate exactly matches all original rational endpoints',compared==12)
    x,y,w=cases[0]
    yy=np.stack([y,1-y],axis=1);ww=np.stack([w,-w],axis=1)
    a=multi.certify_multioutput(x,yy,.03,ww);b=fast.certify_multioutput(x,yy,.03,ww)
    fields=('parameter_error_frobenius_squared_upper','parameter_error_frobenius_upper',
            'objective_gap_upper','meets_parameter_tolerance','work')
    check('dyadic multioutput Frobenius certificate exactly matches original',all(a[k]==b[k] for k in fields))
    rejects('dyadic coordinate cap stays unchanged',lambda:fast.certify_multioutput(x,yy,.03,ww,max_coordinates=11),
            exceptions=(original.CertificateBudgetError,))
    check('dyadic tolerance remains one over 100 million',b['parameter_tolerance']=={'numerator':'1','denominator':'100000000'})
    return {'exact_scalar_endpoint_comparisons':compared,'exact_multioutput_comparisons':1,
            'included_empty_negative_subnormal_and_fractional_inputs':True,
            'new_empirical_dataset':False,'tolerance':1e-8}


def fp32_variant_checks(temporary):
    from phase6 import methods
    from phase3.reference_graph import build_reference_graph
    ids=['p'+str(i) for i in range(5)];sources=['s0','s0','s1','s2','s2']
    curator=np.asarray([[1,0],[1,0],[0,1],[0,1],[1,-1]],np.float32)
    x=np.asarray([[1,.5],[.25,-.5],[-.25,.75],[.5,-1],[.75,.25]],np.float32)
    y=np.asarray([[.25,1],[.5,0],[.75,1],[1,0],[0,1]],np.float64)
    graph=build_reference_graph(curator,ids,.6,sources)
    blockers=graph.blockers;initial={i for i,b in enumerate(blockers) if len(b)==0}
    tested=0
    for unit,owners in [('record',ids),('source',sources)]:
        universe=sorted(set(owners))
        for size in range(min(3,len(universe))+1):
            for dead in itertools.combinations(universe,size):
                selected={i for i,b in enumerate(blockers) if owners[i] not in dead and
                          all(owners[j] in dead for j in b)}
                for method in methods.FP32_METHODS:
                    state=methods.build_method(method,graph,x,y,3,unit=unit,curator=curator)
                    state.delete(dead);moments=state.moments()
                    wanted=sorted(initial-{i for i,o in enumerate(owners) if o in dead}) if method=='B-F-FP32' else sorted(selected)
                    expected_gram=x[wanted].T@x[wanted]
                    expected_cross=x[wanted].T@y[wanted].astype(np.float32)
                    if moments.count!=len(wanted) or not np.array_equal(moments.gram,expected_gram) or not np.array_equal(moments.cross,expected_cross):
                        raise AssertionError(('FP32 variant dyadic moments',method,unit,dead))
                    saved=temporary/('allfp32_'+str(tested)+'.npz');state.snapshot(saved)
                    restored=methods.load_method(method,saved)
                    m2=restored.moments()
                    if not (m2.count==moments.count and np.array_equal(m2.gram,moments.gram) and np.array_equal(m2.cross,moments.cross)):
                        raise AssertionError(('FP32 restored moments',method,unit,dead))
                    if method in ('P-I-FP32','P-S-FP32','P-R-FP32'):
                        if any(v.dtype!=np.float32 for v in restored.coefficients().values()):
                            raise AssertionError('FP32 coefficient state silently promoted')
                    tested+=1
    check('all eight FP32 variants match own targets on dyadic states',tested==272)
    check('all eight FP32 variants preserve full-delete and resume behavior',True)
    check('FP32 summary state and reconstruction remain FP32',True)
    return {'variant_count':len(methods.FP32_METHODS),'algebraic_states':tested,
            'source_and_record_services':True,'actual_semantic_or_source_evidence':False}


def worker_artifact_checks():
    # The complete historical run is stored in a lossless checked archive.
    # Restore it only when absent. A changed destination is never overwritten.
    from tools.phase6_result_archive import verify
    missing=not (ROOT/'empirical_execution/phase6/results/worker_engineering_final/verification.json').is_file()
    restored=verify('worker_engineering_final',restore=missing)
    check('archived worker results bind every member checksum',restored['all_bytes_verified'])
    from ccu.data import read_natural_jsonl,lexical_engineering_features
    from phase4.execution import independent_edges
    result=read(ROOT/'empirical_execution/phase6/results/worker_engineering_final/verification.json')
    for name,value in result['source_sha256'].items():
        path=(ROOT/'empirical_execution/phase6/results/worker_evaluator_before_target_fix.txt'
              if name=='check_workers.py' else ROOT/'empirical_execution/phase6'/name)
        check('worker result source binding '+name,sha(path)==value)
    correction=read(ROOT/'empirical_execution/phase6/results/worker_head_target_recheck.json')
    check('worker evaluator correction remains separate and source bound',correction['all_passed'] and
          correction['old_result_sha256']==sha(ROOT/'empirical_execution/phase6/results/worker_engineering_final/verification.json') and
          correction['corrected_evaluator_sha256']==sha(ROOT/'empirical_execution/phase6/check_workers.py') and
          correction['recheck_source_sha256']==sha(ROOT/'empirical_execution/phase6/recheck_worker_heads.py'))
    rows=read_natural_jsonl(ROOT/'empirical_execution/data/civil_comments_engineering_preview.jsonl',require_labels=True)[:20]
    ids=[r['record_id'] for r in rows];cx,_=lexical_engineering_features(rows,32)
    y=np.asarray([r['label'] for r in rows],np.float64)[:,None]
    _,initial=independent_edges(cx,ids,.6,set())
    requests=[[ids[0]],[ids[1],ids[2]]]
    dimensions={};checked=0;maximum_fp64=maximum_fp32=0.
    for case in result['cases']:
        d=case['dimension']
        if d not in dimensions:dimensions[d]=lexical_engineering_features(rows,d)[0]
        x=dimensions[d]
        service_path=ROOT/case['service_report'];folder=service_path.parent
        service=read(service_path);construction=read(folder/'construction/construction_report.json')
        repair=read(folder/'repair/repair_report.json')
        check('worker real kernel denial '+case['case'],repair['isolation']['denied_existing_file_probes']>0 and
              repair['isolation']['input_file_descriptors_closed_before_repair'])
        initial_head=np.load(folder/'construction/initial_head.npy',allow_pickle=False)
        dead=set()
        heads=[(initial_head,set(initial),construction['head_sha256'],folder/'construction/initial_head.npy')]
        for i,batch in enumerate(requests):
            dead.update(batch);_,selected=independent_edges(cx,ids,.6,dead)
            wanted={j for j in initial if ids[j] not in dead} if case['method'] in ('B-F','B-F-FP32') else set(selected)
            saved=folder/f'repair/head_{i:04d}.npy'
            heads.append((np.load(saved,allow_pickle=False),wanted,repair['releases'][i]['head_sha256'],saved))
        for candidate,selected,binding,path in heads:
            z=x[sorted(selected)].astype(np.float64);yy=y[sorted(selected)]
            expected=z.T@np.linalg.solve(z@z.T+.01*len(z)*np.eye(len(z)),yy) if len(z) else np.zeros_like(candidate)
            err=float(np.linalg.norm(candidate-expected))
            if not np.isfinite(candidate).all() or sha(path)!=binding:
                raise AssertionError(('worker saved head bytes',case['case']))
            if case['method'].endswith('-FP32'):maximum_fp32=max(maximum_fp32,err)
            else:
                maximum_fp64=max(maximum_fp64,err)
                if err>1e-9*max(1.,float(np.linalg.norm(expected))):
                    raise AssertionError(('worker independent saved head',case['case'],err))
            checked+=1
        for release in repair['releases']:
            gate=release['numerical_decoder']
            if not gate['release_allowed'] or gate['normalized_residual_eta']>1e-10:
                raise AssertionError('worker common release gate missing or failed')
            if release['logical_update_meter']['hardware_memory_traffic_bytes'] is not None:
                raise AssertionError('physical memory traffic fabricated')
    check('all saved worker heads independently rebuilt',checked==3*len(result['cases']))
    check('all saved worker common gates preserve tolerance',True)
    return {'services':len(result['cases']),'independent_heads_including_initial':checked,
            'maximum_FP64_head_error_frobenius':maximum_fp64,
            'maximum_FP32_frontier_head_error_frobenius':maximum_fp32,
            'semantic_evidence':False,'natural_source_evidence':False,
            'result_sha256':sha(ROOT/'empirical_execution/phase6/results/worker_engineering_final/verification.json')}


def acceptance_checks(temporary):
    from phase6 import acceptance as a
    from phase3 import panels
    from phase4 import adapters
    design=read(ROOT/'output/empirical_program/study_design.json')
    for panel,target,pool in [('primary-10000',10000,'primary'),('primary-200000',200000,'primary'),
                              ('second-disjoint-10000',10000,'replication'),('separate-refit-5000',5000,'separate_refit')]:
        contract=a.panel_contract({'panel':panel},{'dataset_id':'civil_comments'})
        check('acceptance registry panel '+panel,contract['target']==target and contract['pool']==pool and not contract['calendar'])
    check('News calendar is a separate complete-window target',
          a.panel_contract({'panel':'complete-calendar-window'},{'dataset_id':'cc_news'})==
          {'target':10000,'pool':'calendar','calendar':True})
    rejects('Civil cannot enter News calendar contract',lambda:a.panel_contract({'panel':'complete-calendar-window'},{'dataset_id':'civil_comments'}))
    rejects('smaller caller panel cannot stand for primary target',lambda:a.panel_contract({'panel':'primary-20'},{'dataset_id':'civil_comments'}))
    natural=ROOT/'empirical_execution/data/civil_comments_engineering_preview.jsonl'
    directory=temporary/'source';adapters.adapt_civil(natural,directory,input_schema='engineering_preview')
    spec={'dataset_id':'civil_comments','adapter':{'directory':str(directory),'inputs':{'records':str(natural)},
            'kwargs':{'input_schema':'engineering_preview'}}}
    rows,links,replayed=a.replay_adapter(spec)
    check('source replay preserves all 100 real preview records',len(rows)==100 and links==[] and replayed['all_derived_rows_replayed'])
    refused=a.accept_source_cache({'group_id':'independent','corpus':'civil_comments'}, {'source_acceptance':spec},design)
    check('source preview cannot gain primary acceptance by successful parser replay',
          not refused['machine_acceptance_passed'] and not refused['execution_allowed'])
    check('source replay does not fabricate authenticity',not refused['source_authenticity_verified'] and not refused['human_authenticity_verified'])
    # This is a pure ownership fixture. Its rows are never exported as corpus data.
    fixture=[{'record_id':f'u{i}r{j}','source_unit_id':f'u{i}','partition':'train'}
             for i in range(20) for j in range(i%3+1)]
    contract={'target':4,'exclude_primary_target':5,'exclude_replication_target':3,'source_order_salt':'fixture'}
    chosen,report=a.separate_refit_panel(fixture,design,contract)
    first,_=panels.select_panels(fixture,design,target=5);second,_=panels.select_panels(fixture,design,target=3)
    lookup={r['record_id']:r['source_unit_id'] for r in fixture}
    excluded={lookup[r] for r in set(first['primary'])|set(second['replication'])}
    kept={lookup[r] for r in chosen['separate_refit']}
    check('separate refit excludes complete main and replication sources',not kept&excluded)
    check('separate refit includes each chosen source completely',
          set(chosen['separate_refit'])=={r['record_id'] for r in fixture if r['source_unit_id'] in kept})
    large=a.panel_contract({'panel':'separate-refit-5000'},{'dataset_id':'civil_comments'})
    empty,report=a.separate_refit_panel(fixture,design,large)
    check('separate refit retains real shortfall instead of filling excluded sources',
          not empty['separate_refit'] and report['shortfall']==5000)
    raw=np.asarray([[1,0],[0,1],[.5,.5]],np.float32)
    input_rows=[{'record_id':'a'},{'record_id':'b'},{'record_id':'c'}]
    dossier={'records':[input_rows[2]],'features':raw[2:].copy(),'provenance':{'scope':'software_only'}}
    learner_dossier={**dossier,'provenance':{**dossier['provenance'],'feature_contract':'native_normalized_fp32'}}
    bundle={'curators':{'e5':{'encoder_id':a.emb.E5,'train_ids':['a'],'train':raw[:1].copy(),'calibration':dossier}},
            'learners':{'e5':{'encoder_id':a.emb.E5,'train_ids':['a'],'train':raw[:1].copy(),
                        'evaluation_ids':['b'],'evaluation':raw[1:2].copy(),'calibration':learner_dossier}}}
    data={'e5':(dossier['records'],raw[2:].copy(),dossier['provenance'])}
    a._bind_entries(bundle,{'e5':raw},input_rows,['a'],['b'],data)
    check('source cache rows can pass a correctly aligned software binding',True)
    bundle['learners']['e5']['evaluation']=raw[:1].copy()
    rejects('source cache cannot swap held-out rows',lambda:a._bind_entries(bundle,{'e5':raw},input_rows,['a'],['b'],data))
    return {'natural_preview_records_replayed':100,'authentic_primary_acceptance':False,
            'positive_binding_fixture_is_semantic_data':False,
            'fresh_real_transformer_execution_verified':False}


def dispatcher_checks(temporary):
    from phase6 import dispatch as d
    from phase6.check_dispatch import fixture
    from phase4.execution import independent_edges
    registry,jobs,bundles,design=fixture()
    bundle=next(iter(bundles.values()));group=registry['groups'][0]
    accepted=d.acceptance(group,bundle,design,mode=d.ENGINEERING_MODE,base_dir=ROOT)
    check('dispatcher actual lexical inputs can activate engineering execution',accepted['execution_allowed'])
    check('dispatcher engineering acceptance cannot activate primary',not accepted['primary_inputs_accepted'])
    bad=copy.deepcopy(bundle);bad['natural_input_sha256']='0'*64
    refused=d.acceptance(group,bad,design,mode=d.ENGINEERING_MODE,base_dir=ROOT)
    check('dispatcher changed natural input hash refuses execution',not refused['execution_allowed'])
    graph=d._base_graph(bundle);cell=d.resolve(group,bundle);ids=list(graph.record_ids)
    states=[set(),{0},{1,7},{0,1,7,10},set(range(19)),set(range(20))]
    for index,dead in enumerate(states):
        _,selected=independent_edges(cell['cx'],ids,graph.threshold,[ids[i] for i in dead],seed=0)
        check('dispatcher cached independent graph target '+str(index),d.independent_selection(cell,dead)==set(selected))
    for mutate in ('order','seed'):
        bad=copy.deepcopy(bundle)
        if mutate=='order':bad['requests']['trajectories'][0]['deletion_order'].reverse()
        else:bad['requests']['configuration']['master_seed']+=1
        rejects('dispatcher changed request '+mutate+' refused',lambda:d.recompute_requests(bad,graph))
    primary_design=read(ROOT/'output/empirical_program/study_design.json')
    refused=d.acceptance(group,bundle,primary_design,mode=d.PRIMARY_MODE,base_dir=ROOT)
    checks={row['name']:row for row in refused['checks']}
    check('dispatcher primary requires authoritative complete schedule',not checks['authoritative_primary_request_schedule']['passed'])
    rejects('dispatcher caller cannot invent a primary registry',
            lambda:d.run_dispatch(registry,jobs,bundles,temporary/'forged-primary',mode=d.PRIMARY_MODE))
    path=copy.deepcopy(bundle['requests']['trajectories'][0])
    path['checkpoints']=path['checkpoints'][:-1]
    rejects('dispatcher cannot truncate registered checkpoints',lambda:d._check_path(
        {'arm':path['arm'],'planned_checkpoint_units':[1,2,4]},path,bundle))
    gate=d.saved_head_gate(np.array([[1]],np.float32),np.array([[0]],np.float64),{0},.01,np.array([[1e-11]],np.float64))
    check('dispatcher actual saved head residual controls release',not gate['passed'] and gate['head_frobenius_error']<1e-9)
    check('dispatcher empty target demands exact zero',
        d.saved_head_gate(cell['x'],cell['y'],set(),.01,np.zeros((32,1)))['passed'] and
        not d.saved_head_gate(cell['x'],cell['y'],set(),.01,np.full((32,1),1e-100))['passed'])
    # Exercise real structural routing and unavailable cells without launching
    # another redundant worker benchmark. All inputs are existing Civil text.
    groups=[registry['groups'][0],registry['groups'][-1]]
    gjobs=[j for j in jobs if j['group_id'] in {g['group_id'] for g in groups}]
    small={**registry,'groups':groups,'jobs_content_sha256':d.digest(gjobs)}
    result=d.run_dispatch(small,gjobs,bundles,temporary/'dispatch',design=design)
    ledger=read(temporary/'dispatch/jobs.json')
    check('dispatcher retains every structural and unavailable cell',len(ledger)==len(gjobs) and result['every_planned_job_retained'])
    check('dispatcher unknown source arm stays explicitly unavailable',
          any(r['status']=='structural_zero_or_unavailable_arm' and r['job_id'].endswith('/S000') for r in ledger))
    check('dispatcher missing encoder is preserved as a blocked cell',
          all(r['status']!='completed' for r in ledger if '/missing-e5/' in r['job_id']))
    check('dispatcher retains every requested method in unavailable cells',
          all([m['method'] for m in r['methods']]==next(j['methods'] for j in gjobs if j['job_id']==r['job_id']) for r in ledger))
    check('dispatcher structural execution does not promote preview evidence',result['primary_outputs_accepted']==0)
    return {'natural_rows':20,'direct_graph_states':len(states),'temporary_structural_jobs':len(gjobs),
            'primary_inputs_accepted':False,'real_backend_positive_path_executed':False}


def certificate_artifact_checks():
    base=ROOT/'empirical_execution/phase6/results';total=0;bound={}
    reports=['dyadic_convex_release/checks.json','dyadic_dense_release/checks.json',
             'dyadic_dense_scalar_release/checks.json','human_inference_checks.json','logistic_selection_checks.json']
    for rel in reports:
        p=base/rel;report=read(p)
        check('producer mathematical report passed '+rel,report['status']=='passed' and all(report['checks'].values()))
        for name,value in report['source_sha256'].items():
            candidates=[ROOT/'empirical_execution'/phase/name for phase in ('phase6','phase5','phase4')]
            source=next((q for q in candidates if q.is_file()),None)
            if name=='dyadic_convex.py' and source is not None and sha(source)!=value:
                source=ROOT/'empirical_execution/phase6/history/dyadic_metadata_v1/dyadic_convex.py'
            if source is None or sha(source)!=value:raise AssertionError(('stale producer certificate source',rel,name))
        check('producer mathematical report sources preserved and bound '+rel,True)
        if rel.startswith('dyadic_'):
            runs=read(p.parent/'runs.json');count=len(runs)
            if isinstance(runs,dict):
                raise AssertionError('unexpected certificate run ledger schema')
            for run in runs:
                report_path=p.parent/run['report']
                if not report_path.exists():report_path=ROOT/run['report']
                if run['status']!='completed' or sha(report_path)!=run['report_sha256']:
                    raise AssertionError(('certificate run artifact',rel,run['sequence']))
            total+=count
            check('exact certificate run ledger preserves planned count '+rel,count==(20 if 'convex_release' in rel else 10))
        bound[rel]=sha(p)
    check('all forty exact certificate engineering runs remain recorded',total==40)
    erratum=read(base/'dyadic_metadata_erratum.json')
    check('dyadic memory metadata erratum passed without changing arithmetic',
          erratum['status']=='passed' and all(erratum['checks'].values()) and not erratum['arithmetic_or_sigmoid_changed'])
    check('dyadic corrected and historical source bytes remain separately bound',
          sha(ROOT/'empirical_execution/phase6/dyadic_convex.py')==erratum['corrected_source_sha256'] and
          sha(ROOT/'empirical_execution'/erratum['historical_source_path'])==erratum['historical_source_sha256'] and
          sha(ROOT/'empirical_execution/phase6/check_dyadic_metadata_erratum.py')==erratum['focused_check_sha256'])
    bound['dyadic_metadata_erratum.json']=sha(base/'dyadic_metadata_erratum.json')
    return {'producer_report_sha256':bound,'exact_verifier_fresh_process_runs':total,
            'primary_native_scale_executed':False,'genuine_human_responses_created':0}


def acceptance_result_checks():
    base=ROOT/'empirical_execution/phase6';frozen=read(base/'results/acceptance_freeze.json')
    for name,value in frozen['files'].items():
        check('acceptance final source or result hash '+name,sha(base/name)==value)
    result=read(base/'results/acceptance_checks.json')
    for name,value in result['code_bindings'].items():
        if sha(ROOT/'empirical_execution'/name)!=value:raise AssertionError(('acceptance dependency changed',name))
    check('acceptance all final control and refusal checks passed',result['passed'] and all(r['passed'] for r in result['outcomes']))
    check('acceptance positive controls are explicitly software mocks',
          not result['primary_execution_allowed'] and not result['actual_transformer_cache_replayed'] and
          not result['original_source_rich_archive_accepted'] and result['actual_human_ratings']==0)
    return {'producer_checks':result['checks'],'freeze_sha256':sha(base/'results/acceptance_freeze.json'),
            'positive_control_is_authentic_primary_data':False,'fresh_real_backend_executed':False}


def rank_quotient_checks():
    from phase6.state_audit import rank_basis_relations
    forced={'nodes':[()], 'grounded':False,'omitted_node':(),'stored_basis':[]}
    grounded={**forced,'grounded':True,'omitted_node':None,'stored_basis':[()]}
    nonempty={**forced,'stored_basis':[()]}
    coupled={'nodes':[(),('u',)],'grounded':False,'omitted_node':(),'stored_basis':[('u',)]}
    check('P-R quotient removes only forced-zero zero-dimensional singleton relation',rank_basis_relations([forced])==[])
    check('P-R quotient retains grounded or nonempty coordinate relations',rank_basis_relations([grounded,nonempty,coupled])==[grounded,nonempty,coupled])
    return {'scope':'exact abstract incidence relation qualification; no coordinate tolerance changed',
            'raw_metadata_differences_remain_reported':True}


def final_integration_checks(dispatch_name,extensions_name,state_name,semantics_name):
    """Check complete final ledgers, including failures and absent input cells."""
    from phase6 import recipes
    base=ROOT/'empirical_execution/phase6';results=base/'results'
    for name in (dispatch_name,extensions_name):
        if not (results/name/'checks.json').exists():
            from tools.phase6_result_archive import verify
            verify(name,restore=True)
    dispatch=read(results/dispatch_name/'checks.json')
    check('final dispatcher producer checks passed',dispatch['status']=='passed')
    check('final dispatcher source bytes are current',sha(base/'dispatch.py')==dispatch['dispatch_sha256'] and
          sha(base/'check_dispatch.py')==dispatch['check_sha256'])
    check('final dispatcher preserves all engineering jobs and source stability',
          dispatch['engineering_summary']['every_planned_job_retained'] and
          dispatch['engineering_summary']['inputs_unchanged'] and dispatch['engineering_summary']['source_hashes_unchanged'])
    check('final dispatcher does not create primary evidence',
          dispatch['engineering_summary']['primary_outputs_accepted']==0 and
          dispatch['human_responses_created']==0 and not dispatch['semantic_vectors_created'])
    engineering=read(results/dispatch_name/'run/jobs.json')
    light_instances=0
    for outcome in engineering:
        for method in outcome.get('methods',[]):
            if method.get('requested_precision')=='FP64' and method['method'] in ('B-E','P-I','P-S','P-R'):
                if method['status']!='completed':raise AssertionError('required FP64 method did not complete engineering qualification')
                state=method.get('state_audit',{})
                if not state.get('passed') or len(state['checkpoints'])!=len(method['head_checks']):
                    raise AssertionError('required complete per-release moment audit is absent')
                for row in state['checkpoints']:
                    moments=row['moments_and_membership']
                    if not (row['remaining_horizon_exact'] and moments['count_exact'] and moments['gram']['passed'] and moments['cross']['passed']):
                        raise AssertionError('required complete per-release moment audit failed')
                    if method['method']=='B-E' and not moments['eligible_membership_exact']:
                        raise AssertionError('required B-E membership comparison failed')
                light_instances+=1
    check('every executed FP64 maintenance service compares all moments and required memberships',light_instances>0)
    audited=[r for r in engineering if r['group_id'].startswith('C_full_methods/') and r['job_id'].endswith('/R000')]
    check('final dispatcher exposes one complete eight-method state audit',
          len(audited)==1 and len(audited[0]['methods'])==8 and all(m.get('state_audit',{}).get('passed') for m in audited[0]['methods']))
    state_checkpoints=0
    for method in audited[0]['methods']:
        state=method['state_audit'];sealed=dict(state);binding=sealed.pop('sha256')
        expected_binding=hashlib.sha256(json.dumps(sealed,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        if expected_binding!=binding or state['code_sha256']!=sha(base/'state_audit.py'):
            raise AssertionError(('changed state audit source or result',method['method']))
        if not state['initial_snapshot_matches_measured'] or not state['final_snapshot_matches_measured']:
            raise AssertionError('state audit service endpoints differ from measured service')
        for row in state['checkpoints']:
            required=('reconstructed_actual_snapshot_bound','replay_and_measured_heads_bitwise_equal',
                      'remaining_horizon_exact','alive_units_exact','canonical_record_symbol_state_exact','passed')
            if not all(row[k] for k in required):raise AssertionError(('state audit checkpoint',method['method'],row['index']))
            if method['method'] in ('P-I','P-S','P-R'):
                if not row['coefficients']['complete_union_of_keys_compared'] or not row['coefficients']['passed']:
                    raise AssertionError('state audit omitted full coefficient comparison')
            if method['method']=='P-R' and not (row['rank_basis_relations_exact'] and row['rank_zero_coefficients_exact']):
                raise AssertionError('P-R zero-node quotient lacks exact relation and zero checks')
            state_checkpoints+=1
    check('state audits bind actual auxiliary snapshots at every checkpoint',state_checkpoints==24)
    registry,jobs=recipes.build_registry()
    ledger=read(results/dispatch_name/'missing_primary/jobs.json')
    supplied={r['job_id']:r for r in ledger}
    check('complete missing-primary ledger retains all21332 jobs',len(supplied)==len(ledger)==len(jobs)==21332)
    for job in jobs:
        row=supplied[job['job_id']]
        if row['status']=='completed' or row.get('primary_output_accepted'):
            raise AssertionError(('missing input promoted',job['job_id']))
        if [m['method'] for m in row['methods']]!=job['methods']:
            raise AssertionError(('missing input method omitted',job['job_id']))
    check('all missing-primary methods and explicit statuses remain recorded',True)
    extensions=read(results/extensions_name/'checks.json')
    check('final extension route checks passed',extensions['status']=='passed')
    for name,value in extensions['source_sha256'].items():
        if sha(base/name)!=value:raise AssertionError(('stale final extension source',name))
    check('final extension production sources are current',True)
    check('final extension tests do not create sources human ratings or primary results',
          not extensions['genuine_source_withdrawal_evidence'] and extensions['actual_human_responses']==0 and
          not extensions['primary_semantic_study_started'])
    report=read(results/'recipes_checks_final.json')
    check('final recipe checks passed and bind current source',report['status']=='passed' and
          report['recipe_code_sha256']==sha(base/'recipes.py'))
    check('final recipe release preserves all complete planned cells',
          read(results/'recipes_release/registry.json')==registry and
          read(results/'recipes_release/recipes.json')==recipes.build_recipe_book())
    statistics=read(results/'statistics_evidence_checks_v2.json')
    check('final statistics evidence binding checks passed',statistics['all_passed'] and
          all(c['passed'] for c in statistics['checks']))
    for name,value in statistics['source_sha256'].items():
        if sha(base/name)!=value:raise AssertionError(('stale statistics binding source',name))
    check('final statistics evidence checker binds current source',True)
    check('statistics schema fixtures are not promoted to primary evidence',not statistics['primary_evidence_created'])
    state_result=read(results/state_name/'checks.json')
    check('final eight-method durable state audit checks passed',
          state_result['status']=='passed' and all(state_result['checks'].values()) and state_result['natural_checkpoint_states']==32)
    for name,value in state_result['source_sha256'].items():
        if sha(base/name)!=value:raise AssertionError(('stale state audit source',name))
    check('final durable state audit binds current production source',True)
    check('state audit natural record results do not claim native source trajectories',
          not state_result['primary_semantic_or_genuine_source_evidence'] and
          not state_result['all_16_R_and_16_S_native_paths_completed'])
    semantics=read(results/semantics_name/'checks.json')
    check('all final state transition contracts passed',semantics['all_passed'] and all(c['passed'] for c in semantics['checks']))
    for name,value in semantics['source_sha256'].items():
        if sha(ROOT/'empirical_execution'/name)!=value:raise AssertionError(('stale transition contract source',name))
    check('final transition contracts bind current sources and natural input',True)
    return {'dispatcher_checks':dispatch['check_count'],'extension_checks':extensions['check_count'],
            'recipe_checks':report['check_count'],'statistics_binding_checks':statistics['check_count'],
            'complete_primary_jobs_blocked':len(ledger),'actual_auxiliary_state_audit_checkpoints':state_checkpoints,
            'per_release_moment_audited_engineering_services':light_instances,
            'standalone_natural_state_audit_checkpoints':state_result['natural_checkpoint_states'],
            'state_transition_contract_checks':semantics['check_count'],
            'report_sha256':{str(p.relative_to(results)):sha(p) for p in
                (results/dispatch_name/'checks.json',results/extensions_name/'checks.json',results/'recipes_checks_final.json',
                 results/'statistics_evidence_checks_v2.json',results/state_name/'checks.json',
                 results/semantics_name/'checks.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'empirical_execution/phase6/results/independent_review')
    parser.add_argument('--final',action='store_true',help='Require all frozen integration reports before passing the scoped review')
    parser.add_argument('--dispatch-result',default='dispatch_final')
    parser.add_argument('--extensions-result',default='extensions_final')
    parser.add_argument('--state-result',default='state_audit_release')
    parser.add_argument('--state-semantics-result',default='state_audit_semantics_final')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='phase6-independent-') as temporary:
        temporary = Path(temporary)
        sections = {'human_intervals': human_interval_checks(),
                    'methods_and_measurements': method_checks(temporary),
                    'recipes':recipe_checks(),'logistic_selection':logistic_checks(),
                    'dyadic_certification':dyadic_certificate_checks(),
                    'all_FP32_variants':fp32_variant_checks(temporary),
                    'worker_saved_artifacts':worker_artifact_checks(),
                    'source_and_cache_acceptance':acceptance_checks(temporary),
                    'dispatcher':dispatcher_checks(temporary),
                    'mathematical_result_lineage':certificate_artifact_checks(),
                    'final_acceptance_lineage':acceptance_result_checks(),
                    'P_R_rank_zero_qualification':rank_quotient_checks()}
        if args.final:sections['final_integration']=final_integration_checks(
            args.dispatch_result,args.extensions_result,args.state_result,args.state_semantics_result)
    result = {'schema': 'ccu-phase6-independent-review-1',
              'status': 'passed_scoped_independent_review' if args.final else 'incomplete_review', 'check_count': len(CHECKS),
              'checks': CHECKS, 'sections': sections,
              'reviewed_sources': source_hashes(),
              'primary_semantic_study_completed': False,
              'authentic_data_provenance_verified': False,
              'genuine_human_collection_verified': False,
              'unexecuted_primary_boundaries': ['authentic source-rich corpus intake','complete real E5 and MPNet cache replay',
                  'independent human annotation collection','native-scale confirmatory services and statistics'],
              'scope': 'Independent mathematical and software checks only'}
    (args.output_dir/'review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','reviewed_sources')}))


if __name__ == '__main__':
    main()
