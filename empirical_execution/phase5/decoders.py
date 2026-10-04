"""Prospective common numerical gate and zero-start CG ablation (version 1).

Independent of the frozen Phase 5 worker run. No audit resets a failed service
state or replaces a failed head. Diagnostics are not interval certificates.
"""
from dataclasses import dataclass
import hashlib,math,time
import numpy as np
from scipy.linalg import cho_factor,cho_solve

SCHEMA='ccu-common-ridge-decoder-1'
DEFAULT_POLICY={'schema':SCHEMA,'normalized_residual_limit':1e-10,
                'cg_max_iterations_per_dimension':10,'cg_initialization':'zero',
                'residual_accumulation':'coordinate_order_Neumaier_FP64',
                'empty_target_policy':'reject_nonzero_moments',
                'failure_policy':'retain_failure_no_fallback_no_tolerance_relaxation'}

@dataclass
class DecodeResult:
    weights: np.ndarray | None
    diagnostics: dict

def _hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes()).hexdigest()

def _norm(a):
    """Stable ordinary floating Frobenius norm; not an outward-rounded bound."""
    return math.hypot(*np.asarray(a).reshape(-1).tolist())

def _compensated_product(a,b):
    """Coordinate-ordered Neumaier accumulation of ordinary rounded products."""
    a=np.asarray(a);b=np.asarray(b);dtype=np.result_type(a.dtype,b.dtype)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError('Aligned matrices required')
    total=np.zeros((a.shape[0],b.shape[1]),dtype=dtype);correction=np.zeros_like(total)
    with np.errstate(over='raise',invalid='raise'):
        for k in range(a.shape[1]):
            term=a[:,k:k+1]*b[k:k+1,:];nxt=total+term
            correction+=np.where(np.abs(total)>=np.abs(term),(total-nxt)+term,(term-nxt)+total)
            total=nxt
        return total+correction

def _residual(a,w,h,opnorm):
    r=_compensated_product(a,w)-h
    if not np.isfinite(r).all():raise FloatingPointError('Nonfinite residual')
    numerator=_norm(r);wnorm=_norm(w);hnorm=_norm(h)
    if not all(math.isfinite(v) for v in (numerator,wnorm,hnorm)):raise FloatingPointError('Residual normalization norm exceeds FP64 range')
    # Extended scalar arithmetic avoids overflowing an otherwise meaningful
    # normalization; it does not turn the residual computation into a bound.
    den=np.longdouble(opnorm)*np.longdouble(wnorm)+np.longdouble(hnorm)
    if den==0:eta=0. if numerator==0 else math.inf
    else:eta=float(np.longdouble(numerator)/den)
    return {'absolute_residual_fro':numerator,'normalized_residual_eta':eta,
            'normalization_denominator_decimal':str(den),'weights_fro':wnorm,'cross_fro':hnorm}

def _validate_policy(policy):
    # This v1 is locked. A later development amendment needs a new schema rather
    # than callers loosening the residual gate or iteration budget after failure.
    if policy!=DEFAULT_POLICY:raise ValueError('Version 1 decoder policy is fixed; version any amendment')

def decode(moments,lambda_reg,*,solver='cholesky',policy=None):
    """Return a candidate plus explicit release gate; never silently recover.

    CG solves output columns separately from zero. Each column checks its true
    compensated residual against the same eta gate after every iteration. The
    combined matrix residual is recomputed before release. No warm start or
    preconditioner is used. Max iterations per column are prospectively 10*d.
    Invalid input is a ValueError; numerical failures are returned and retained.
    """
    started=time.perf_counter();policy=dict(DEFAULT_POLICY if policy is None else policy);_validate_policy(policy)
    if solver not in ('cholesky','cg'):raise ValueError('Unknown common solver')
    if isinstance(lambda_reg,(bool,np.bool_)) or not isinstance(lambda_reg,(int,float,np.integer,np.floating)) or not math.isfinite(float(lambda_reg)) or lambda_reg<=0:raise ValueError('Finite positive scalar lambda required')
    m=np.asarray(moments.gram);h=np.asarray(moments.cross);n=moments.count
    if m.dtype!=np.float64 or h.dtype!=np.float64 or m.ndim!=2 or m.shape[0]!=m.shape[1] or m.shape[0]<1 or h.ndim!=2 or h.shape[0]!=m.shape[0] or h.shape[1]<1:raise ValueError('FP64 square Gram and aligned nonempty-output cross matrix required')
    if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or not 0<=n<2**52:raise ValueError('Exact nonnegative count below2**52 required')
    if not np.isfinite(m).all() or not np.isfinite(h).all():raise ValueError('Finite moments required')
    d,c=h.shape;lam=float(lambda_reg);n=int(n)
    info={'schema':SCHEMA,'solver':solver,'policy':policy,'count':n,'dimension':d,'outputs':c,'lambda_hex':lam.hex(),
          'gram_sha256':_hash(m),'cross_sha256':_hash(h),'release_allowed':False,'numerically_certified':False,
          'parameter_error_bound_is_diagnostic':True,'fallback_performed':False,'audit_requested':False,
          'condition_number_2_estimate':None,'condition_log10_estimate':None,'iteration_counts':[]}
    def fail(status,weights=None,exception=None):
        info.update(status=status,audit_requested=True,total_seconds=time.perf_counter()-started)
        if exception is not None:info['exception']=str(exception)
        return DecodeResult(weights,info)
    if not np.array_equal(m,m.T):return fail('invalid_asymmetric_stored_gram')
    if n==0:
        info['empty_gram_max_abs']=float(np.max(np.abs(m),initial=0));info['empty_cross_max_abs']=float(np.max(np.abs(h),initial=0))
        if np.any(m!=0) or np.any(h!=0):return fail('empty_count_nonzero_moments')
        w=np.zeros_like(h);info.update(status='passed_empty_target',release_allowed=True,normalized_residual_eta=0.,absolute_residual_fro=0.,
                                      weights_sha256=_hash(w),total_seconds=time.perf_counter()-started,condition_scope='undefined_empty_objective')
        return DecodeResult(w,info)
    try:
        formed=time.perf_counter();a=m.copy()
        with np.errstate(over='raise',invalid='raise'):
            shift=np.float64(lam)*np.float64(n);a.flat[::d+1]+=shift
        info['system_formation_seconds']=time.perf_counter()-formed
        condstart=time.perf_counter();eigenvalues=np.linalg.eigvalsh(a)
        if not np.isfinite(eigenvalues).all():return fail('nonfinite_condition_estimate')
        low,high=float(eigenvalues[0]),float(eigenvalues[-1]);opnorm=max(abs(low),abs(high))
        info.update(matrix_norm_2_estimate=opnorm,minimum_eigenvalue_estimate=low,maximum_eigenvalue_estimate=high,
                    conditioning_seconds=time.perf_counter()-condstart,gram_minimum_eigenvalue_shift_estimate=low-float(shift),
                    conditioning_estimator='FP64_symmetric_eigvalsh_not_certificate')
        if low<=0:return fail('nonpositive_system_eigenvalue_estimate')
        logcond=math.log10(high)-math.log10(low);ratio=np.longdouble(high)/np.longdouble(low)
        info['condition_log10_estimate']=logcond;info['condition_number_2_estimate']=float(ratio) if ratio<=np.finfo(np.float64).max else None
        solve_start=time.perf_counter()
        if solver=='cholesky':
            factor=cho_factor(a,lower=True,overwrite_a=False,check_finite=True)
            w=cho_solve(factor,h,overwrite_b=False,check_finite=True)
        else:
            w=np.zeros_like(h);maxiter=policy['cg_max_iterations_per_dimension']*d
            for j in range(c):
                b=h[:,j];r=b.copy();p=r.copy();rs=math.fsum(float(v)*float(v) for v in r)
                if rs==0:
                    # A nonzero subnormal RHS must not silently become an empty
                    # CG problem when its squared norm underflows.
                    if np.any(b!=0):return fail('cg_squared_norm_underflow',w)
                    info['iteration_counts'].append(0);continue
                passed=False
                for iteration in range(1,maxiter+1):
                    ap=_compensated_product(a,p[:,None])[:,0]
                    pap=math.fsum(float(u)*float(v) for u,v in zip(p,ap))
                    if not math.isfinite(pap) or pap<=0:return fail('cg_nonpositive_or_nonfinite_curvature',w)
                    alpha=rs/pap;w[:,j]+=alpha*p;r-=alpha*ap
                    truth=_residual(a,w[:,j:j+1],h[:,j:j+1],opnorm)
                    if truth['normalized_residual_eta']<=policy['normalized_residual_limit']:
                        passed=True;info['iteration_counts'].append(iteration);break
                    next_rs=math.fsum(float(v)*float(v) for v in r)
                    if not math.isfinite(next_rs) or next_rs<=0:return fail('cg_residual_breakdown',w)
                    p=r+(next_rs/rs)*p;rs=next_rs
                if not passed:info['iteration_counts'].append(maxiter);return fail('cg_iteration_limit',w)
        info['solve_seconds']=time.perf_counter()-solve_start
        if not np.isfinite(w).all():return fail('nonfinite_solution',w)
        residual_start=time.perf_counter();info.update(_residual(a,w,h,opnorm));info['residual_seconds']=time.perf_counter()-residual_start
        bound=np.longdouble(info['absolute_residual_fro'])/np.longdouble(shift)
        info['parameter_error_from_stored_moments_diagnostic']=float(bound) if bound<=np.finfo(np.float64).max else None
        info['parameter_error_from_stored_moments_diagnostic_decimal']=str(bound)
        info['parameter_diagnostic_assumption']='Exact target Gram PSD and no moment accumulation error; not an original-data certificate'
        info['weights_sha256']=_hash(w)
        if not math.isfinite(info['normalized_residual_eta']) or info['normalized_residual_eta']>policy['normalized_residual_limit']:return fail('normalized_residual_gate_failed',w)
        info.update(status='passed_residual_verified',release_allowed=True,total_seconds=time.perf_counter()-started)
        return DecodeResult(w,info)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError,ValueError) as exc:
        return fail('numerical_exception',exception=exc)

def extended_audit(features,targets,candidate,lambda_reg,*,computed_moments=None,reason='preselected_audit'):
    """Evaluator-only retained-row audit, never a service repair/fallback.

    Rebuilds moments and residual independently in genuine NumPy long double,
    with compensated record/coordinate accumulation. Requires wider mantissa
    than FP64; unsupported platforms fail explicitly. Does not certify intervals.
    """
    started=time.perf_counter()
    if reason not in ('preselected_audit','numerical_failure'):raise ValueError('Audit selection reason required')
    if np.finfo(np.longdouble).nmant<=np.finfo(np.float64).nmant:raise RuntimeError('Genuinely extended floating mantissa unavailable')
    x=np.asarray(features);y=np.asarray(targets);w=np.asarray(candidate)
    if y.ndim==1:y=y[:,None]
    if x.dtype!=np.float32 or y.dtype!=np.float64 or w.dtype!=np.float64 or x.ndim!=2 or y.ndim!=2 or w.ndim!=2 or len(y)!=len(x) or w.shape!=(x.shape[1],y.shape[1]):raise ValueError('Aligned canonical FP32 rows and FP64 targets/head required')
    if not all(np.isfinite(a).all() for a in (x,y,w)) or not math.isfinite(float(lambda_reg)) or lambda_reg<=0:raise ValueError('Finite values and positive lambda required')
    n,d=x.shape;c=y.shape[1];dtype=np.longdouble;gx=np.zeros((d,d),dtype);gc=np.zeros_like(gx);hy=np.zeros((d,c),dtype);hc=np.zeros_like(hy)
    def add(total,correction,term):
        nxt=total+term;correction+=np.where(np.abs(total)>=np.abs(term),(total-nxt)+term,(term-nxt)+total);total[:]=nxt
    with np.errstate(over='raise',invalid='raise'):
        for row,target in zip(x,y):
            xx=row.astype(dtype);yy=target.astype(dtype);add(gx,gc,xx[:,None]*xx[None,:]);add(hy,hc,xx[:,None]*yy[None,:])
        gx+=gc;hy+=hc;a=gx.copy();a.flat[::d+1]+=dtype(float(lambda_reg))*dtype(n)
        residual=_compensated_product(a,w.astype(dtype))-hy
        norm=lambda value:np.sqrt(np.sum(np.square(value),dtype=dtype))
        rnorm=norm(residual);scale=dtype(float(lambda_reg))*dtype(n)
        error=norm(w.astype(dtype)) if n==0 else rnorm/scale
    report={'schema':'ccu-extended-ridge-audit-1','reason':reason,'count':n,'dimension':d,'outputs':c,
            'longdouble_mantissa_bits_including_hidden':int(np.finfo(dtype).nmant)+1,
            'residual_fro_decimal':str(rnorm),'parameter_error_diagnostic_decimal':str(error),
            'original_row_features_sha256':_hash(x),'original_targets_sha256':_hash(y),'candidate_sha256':_hash(w),
            'retained_row_access':'independent_evaluator_only','numerically_certified':False,
            'service_failure_overridden':False,'replacement_head_produced':False,
            'array_workspace_lower_bytes':int(gx.nbytes+gc.nbytes+hy.nbytes+hc.nbytes+a.nbytes+residual.nbytes),
            'rounding_error_interval_available':False,'audit_seconds':time.perf_counter()-started}
    if computed_moments is not None:
        mm=np.asarray(computed_moments.gram,dtype);hh=np.asarray(computed_moments.cross,dtype)
        if mm.shape!=gx.shape or hh.shape!=hy.shape:raise ValueError('Computed moments shape mismatch')
        report['computed_count_equal']=computed_moments.count==n
        report['gram_error_fro_diagnostic_decimal']=str(norm(mm-gx))
        report['cross_error_fro_diagnostic_decimal']=str(norm(hh-hy))
    return report
