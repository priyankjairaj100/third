"""Independent fractional/binary logistic outputs; frozen scalar verifier reuse.

Objective: sum_c mean_i BCE(y_ic, x_i @ W_c) + lambda/2 ||W||_F^2.
All outputs use the same positive lambda, no intercept and no output averaging.
Small software controls are not empirical Stack data. No privacy guarantee.
"""
from pathlib import Path
from fractions import Fraction
import hashlib, json, time
import numpy as np
from phase4 import convex as scalar
from phase4.requests import digest, verify_manifest, _graph_state

SCHEMA='ccu-multioutput-logistic-v1'

def fraction(value):
    return Fraction(int(value['numerator']),int(value['denominator']))

def _inputs(x,y,w=None):
    x=np.asarray(x);y=np.asarray(y)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('Finite FP32 [n,d] features required')
    if y.dtype!=np.float64 or y.ndim!=2 or not y.shape[1] or y.shape[0]!=len(x):
        raise ValueError('Aligned FP64 [n,q] targets with q>=1 required')
    if not np.isfinite(y).all() or np.any(y<0) or np.any(y>1):
        raise ValueError('Every fractional target must be in [0,1]')
    w=np.zeros((x.shape[1],y.shape[1]),dtype=np.float64) if w is None else np.asarray(w)
    if w.dtype!=np.float64 or w.shape!=(x.shape[1],y.shape[1]) or not np.isfinite(w).all():
        raise ValueError('Finite FP64 [d,q] weights required')
    scalar._inputs(x,y[:,0],w[:,0])
    return x,y,w

def objective_gradient(features,targets,weights,lambda_reg):
    x,y,w=_inputs(features,targets,weights)
    evaluations=[scalar.objective_gradient_hessian(x,y[:,c],w[:,c],lambda_reg,hessian=False)
                 for c in range(y.shape[1])]
    return sum(r[0] for r in evaluations),np.column_stack([r[1] for r in evaluations])

def solve_multioutput(features,targets,lambda_reg,*,warm_start=None,**solver_options):
    x,y,w=_inputs(features,targets,warm_start);start=time.perf_counter()
    outputs=[];heads=[]
    for c in range(y.shape[1]):
        result=scalar.solve_fractional_logistic(x,y[:,c],lambda_reg,warm_start=w[:,c],**solver_options)
        heads.append(result.pop('weights'));outputs.append(result)
    counters=('joint_objective_gradient_hessian_evaluations','line_search_objective_evaluations',
      'resolution_gradient_evaluations','resolution_gradient_progress_steps','forward_rows','gradient_rows',
      'hessian_rows','objective_rows','linear_solves','accepted_steps','backtracks')
    meter={key:sum(r['meter'][key] for r in outputs) for key in counters}
    meter.update(counter_units='row counters count output-row evaluations; outputs are solved sequentially',
      shared_fp32_feature_bytes=int(x.nbytes),shared_fp64_target_bytes=int(y.nbytes),
      output_fp64_head_bytes=int(w.nbytes),sequential_promoted_feature_bytes=8*x.size,
      sequential_hessian_bytes=8*x.shape[1]**2,peak_rss_measured=False,
      complete_retained_payload_queries_charged=True)
    return dict(schema=SCHEMA,weights=np.column_stack(heads),objective=sum(r['objective'] for r in outputs),
      output_results=outputs,meter=meter,elapsed_seconds=time.perf_counter()-start,
      objective_normalization='sum_over_outputs_mean_over_rows',outputs=y.shape[1])

def certify_multioutput(features,targets,lambda_reg,weights,*,max_coordinates=2_000_000,
                        parameter_tolerance=Fraction(1,10**8),**certificate_options):
    x,y,w=_inputs(features,targets,weights);start=time.perf_counter()
    cap=scalar._integer(max_coordinates,'max_coordinates',0)
    if x.size*y.shape[1]>cap:
        raise scalar.CertificateBudgetError('Total output-row-coordinate budget exceeded before verification')
    tolerance=scalar.rational(parameter_tolerance)
    if tolerance<0:raise ValueError('nonnegative Frobenius tolerance required')
    certificates=[scalar.certify_logistic(x,y[:,c],lambda_reg,w[:,c],max_coordinates=cap,
                    parameter_tolerance=tolerance,**certificate_options) for c in range(y.shape[1])]
    squared=sum((fraction(c['parameter_error_squared_upper']) for c in certificates),Fraction(0))
    gap=sum((fraction(c['objective_gap_upper']) for c in certificates),Fraction(0))
    radius=scalar.sqrt_upper_rational(squared,30)
    work={key:sum(c['work'][key] for c in certificates) for key in certificates[0]['work']}
    result=dict(schema='ccu-multioutput-logistic-certificate-v1',outputs=y.shape[1],records=len(x),
      objective_normalization='sum_over_outputs_mean_over_rows',scalar_certificates=certificates,
      parameter_error_frobenius_squared_upper=scalar._fraction_json(squared),
      parameter_error_frobenius_upper=scalar._fraction_json(radius),
      objective_gap_upper=scalar._fraction_json(gap),parameter_tolerance=scalar._fraction_json(tolerance),
      meets_parameter_tolerance=squared<=tolerance*tolerance,
      radius_float_for_display_only=float(radius),work=work,total_coordinate_budget=cap,
      input_bindings={name:scalar._hash_array(a) for name,a in [('features',x),('targets',y),('weights',w)]},
      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      scalar_code_sha256=hashlib.sha256(Path(scalar.__file__).read_bytes()).hexdigest(),
      elapsed_seconds=time.perf_counter()-start,statistical_unlearning_certificate=False,
      canonical_state_claim=False,privacy_claim=False)
    result['sha256']=digest(result);return result

def signed_gradient(old_gradient,weights,lambda_reg,old_count,new_count,
                    removed_features,removed_targets,added_features,added_targets):
    xr,yr,w=_inputs(removed_features,removed_targets,weights)
    xa,ya,_=_inputs(added_features,added_targets,weights);g=np.asarray(old_gradient)
    if g.dtype!=np.float64 or g.shape!=w.shape or not np.isfinite(g).all():
        raise ValueError('finite FP64 aligned old matrix gradient required')
    return np.column_stack([scalar.signed_warm_gradient(g[:,c],w[:,c],lambda_reg,old_count,new_count,
                xr,yr[:,c],xa,ya[:,c]) for c in range(yr.shape[1])])

def retained_selection(graph,request_manifest,trajectory_id,checkpoint):
    """Resolve RECORD or SOURCE requests with whole-source expansion, never proxies as native."""
    verify_manifest(request_manifest,graph);ids,owners,blockers,_=_graph_state(graph)
    paths=[p for p in request_manifest['trajectories'] if p['trajectory_id']==trajectory_id]
    if len(paths)!=1:raise ValueError('unknown/nonunique trajectory')
    path=paths[0]
    if checkpoint not in path['checkpoints']:raise ValueError('checkpoint is not frozen in manifest')
    prefix=set(path['deletion_order'][:checkpoint])
    dead=prefix if path['unit']=='record' else {rid for rid in ids if owners[rid] in prefix}
    selected={rid for rid in ids if rid not in dead and blockers[rid]<=dead}
    return sorted(selected),sorted(dead)

def save_resume(path,weights,*,selected_ids,deleted_ids,input_bindings):
    """Persist only warm-candidate state. Payload/graph remain required, separately charged."""
    w=np.asarray(weights)
    if w.dtype!=np.float64 or w.ndim!=2 or not np.isfinite(w).all():raise ValueError('FP64 head matrix required')
    selected=sorted(selected_ids);deleted=sorted(deleted_ids)
    if len(set(selected))!=len(selected) or len(set(deleted))!=len(deleted) or set(selected)&set(deleted):
        raise ValueError('unique disjoint selected and deleted IDs required')
    value=dict(schema='ccu-multioutput-warm-resume-v1',weights=w.tolist(),selected_ids=selected,
      deleted_ids=deleted,input_bindings=input_bindings,
      contract='warm candidate only; retained payload and graph required after resume; recertify next release')
    value['sha256']=digest(value);path=Path(path);start=time.perf_counter()
    with path.open('x') as handle:json.dump(value,handle,sort_keys=True,separators=(',',':'),allow_nan=False)
    return dict(serialized_bytes=path.stat().st_size,write_seconds=time.perf_counter()-start,
      payload_included=False,graph_included=False,object_memory_not_serialized_bytes=True)

def load_resume(path,*,input_bindings,expected_selected_ids,expected_deleted_ids):
    start=time.perf_counter();path=Path(path);raw=path.read_bytes();value=json.loads(raw)
    if value.get('schema')!='ccu-multioutput-warm-resume-v1' or value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('resume schema or seal mismatch')
    if value['input_bindings']!=input_bindings or value['selected_ids']!=sorted(expected_selected_ids) or value['deleted_ids']!=sorted(expected_deleted_ids):
        raise ValueError('resume input/request state binding mismatch')
    w=np.asarray(value['weights'],dtype=np.float64)
    if w.ndim!=2 or not np.isfinite(w).all():raise ValueError('invalid saved head')
    return w,dict(serialized_bytes=len(raw),read_seconds=time.perf_counter()-start,
      parsed_head_bytes=w.nbytes,payload_reload_not_included=True,graph_reload_not_included=True)
