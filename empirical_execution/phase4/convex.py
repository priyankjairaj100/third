"""Fractional-label logistic repair and exact residual certificates.

The target is average single-output cross entropy plus lambda/2 * ||w||².
All coordinates are penalized; there is no implicit intercept. Empty data has
zero data loss. Floating Newton outputs are candidates; only the rational
certificate supplies a rigorous optimization-error bound for stored inputs.
This is not a statistical/privacy unlearning guarantee or canonical-state claim.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,math,time
from fractions import Fraction
from numbers import Integral
import numpy as np
from scipy.special import expit
from scipy.linalg import cho_factor,cho_solve
try:
    from ccu.certified_ridge import rational,sqrt_upper_rational
except ImportError:
    from empirical_execution.ccu.certified_ridge import rational,sqrt_upper_rational


class CertificateBudgetError(RuntimeError):pass


def _positive(value,name):
    if isinstance(value,np.floating) and value.dtype.itemsize>8:
        raise ValueError(f'{name} cannot silently narrow extended floating-point input')
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value) or value<=0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _integer(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _inputs(features,targets,weights=None):
    x=np.asarray(features);y=np.asarray(targets)
    if y.ndim==2 and y.shape[1]==1:y=y[:,0]
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]<1 or y.dtype!=np.float64 or y.shape!=(len(x),):
        raise ValueError('Aligned FP32 [n,d] features and FP64 single-output fractional targets required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y<0) or np.any(y>1):
        raise ValueError('Finite features and fractional targets in [0,1] required')
    w=np.zeros(x.shape[1],dtype=np.float64) if weights is None else np.asarray(weights)
    if w.dtype!=np.float64 or w.shape!=(x.shape[1],) or not np.isfinite(w).all():
        raise ValueError('Finite aligned FP64 weight vector required')
    return x,y,w


def objective_gradient_hessian(features,targets,weights,lambda_reg,*,hessian=True):
    """Unmetered numerical audit interface; evaluation uses all supplied rows."""
    x,y,w=_inputs(features,targets,weights);lam=_positive(lambda_reg,'lambda_reg')
    z=x.astype(np.float64);n=len(z);s=z@w
    if not n:
        return .5*lam*float(w@w),lam*w.copy(),lam*np.eye(len(w)) if hessian else None
    # Stable fractional BCE avoids softplus(s)-y*s cancellation for large s.
    loss=np.where(s>=0,np.logaddexp(0,-s)+(1-y)*s,np.logaddexp(0,s)-y*s)
    p=expit(s);g=z.T@(p-y)/n+lam*w
    matrix=(z.T@((p*(1-p))[:,None]*z)/n+lam*np.eye(len(w))) if hessian else None
    return float(np.mean(loss)+.5*lam*(w@w)),g,matrix


def solve_fractional_logistic(features,targets,lambda_reg,*,warm_start=None,
                             gradient_tolerance=1e-11,max_iterations=100,
                             max_backtracks=60):
    """Damped Newton with counted full-retained-data evaluations; no moment trick.

    Both cold and warm candidates use this identical solver/stopping rule.
    A returned 'converged' status is numerical, and must not replace a certificate.
    """
    x,y,w0=_inputs(features,targets,warm_start);lam=_positive(lambda_reg,'lambda_reg')
    tol=_positive(gradient_tolerance,'gradient_tolerance')
    iterations=_integer(max_iterations,'max_iterations');backtracks=_integer(max_backtracks,'max_backtracks')
    w=w0.copy();n,d=x.shape;z=x.astype(np.float64)
    meter={'joint_objective_gradient_hessian_evaluations':0,'line_search_objective_evaluations':0,
      'resolution_gradient_evaluations':0,'resolution_gradient_progress_steps':0,
      'forward_rows':0,'gradient_rows':0,'hessian_rows':0,'objective_rows':0,
      'linear_solves':0,'accepted_steps':0,'backtracks':0,'selected_records':n,
      'feature_fp32_input_bytes':int(x.nbytes),'target_fp64_input_bytes':int(y.nbytes),
      'promoted_feature_workspace_bytes':int(z.nbytes),'candidate_weight_bytes':int(w.nbytes),
      'hessian_matrix_bytes':8*d*d,'peak_rss_measured':False}
    started=time.perf_counter();trace=[]
    def evaluate(v,need_hessian):
        s=z@v;p=expit(s)
        if n:
            loss=np.where(s>=0,np.logaddexp(0,-s)+(1-y)*s,np.logaddexp(0,s)-y*s)
            obj=float(np.mean(loss)+.5*lam*(v@v))
        else:obj=float(.5*lam*(v@v))
        meter['forward_rows']+=n;meter['objective_rows']+=n
        if not need_hessian:
            meter['line_search_objective_evaluations']+=1
            return obj
        meter['joint_objective_gradient_hessian_evaluations']+=1
        meter['gradient_rows']+=n;meter['hessian_rows']+=n
        g=z.T@(p-y)/n+lam*v if n else lam*v
        h=z.T@((p*(1-p))[:,None]*z)/n+lam*np.eye(d) if n else lam*np.eye(d)
        return obj,g,h
    def trial_gradient(v):
        meter['resolution_gradient_evaluations']+=1
        meter['forward_rows']+=n;meter['gradient_rows']+=n
        return z.T@(expit(z@v)-y)/n+lam*v
    # Empty target has a known exact minimizer and requires no data passes.
    if not n:
        return {'weights':np.zeros(d,dtype=np.float64),'status':'empty_target_exact_zero',
          'objective':0.,'gradient_norm_diagnostic':0.,'iterations':0,'trace':[],
          'meter':meter,'elapsed_seconds':time.perf_counter()-started}
    status='iteration_limit';objective=None;gradient_norm=None
    for step in range(iterations+1):
        objective,g,h=evaluate(w,True);gradient_norm=float(np.linalg.norm(g))
        trace.append({'iteration':step,'objective':objective,'gradient_norm_diagnostic':gradient_norm})
        if not math.isfinite(objective) or not np.isfinite(g).all() or not np.isfinite(h).all():
            status='nonfinite_numerical_evaluation';break
        if gradient_norm<=tol:status='numerically_converged';break
        if step==iterations:break
        try:
            delta=cho_solve(cho_factor(h,lower=True,check_finite=True),-g,check_finite=True)
            meter['linear_solves']+=1
        except Exception as error:
            status=f'linear_solve_failed:{type(error).__name__}';break
        directional=float(g@delta)
        if not math.isfinite(directional) or directional>=0:status='non_descent_direction';break
        accepted=False;step_size=1.
        for bt in range(backtracks):
            trial=w+step_size*delta;candidate=evaluate(trial,False)
            if math.isfinite(candidate) and candidate<=objective+1e-4*step_size*directional:
                w=trial;accepted=True;meter['accepted_steps']+=1;break
            # A numerically unresolved objective decrease cannot reliably choose
            # a tiny Newton step. In this narrow regime, demand measurable
            # gradient progress; the final exact certificate still decides fidelity.
            resolution=32*np.finfo(np.float64).eps*max(1.,abs(objective))
            if (math.isfinite(candidate) and -step_size*directional<=resolution
                    and candidate<=objective+resolution):
                trial_g=trial_gradient(trial)
                if np.isfinite(trial_g).all() and np.linalg.norm(trial_g)<=.5*gradient_norm:
                    w=trial;accepted=True;meter['accepted_steps']+=1
                    meter['resolution_gradient_progress_steps']+=1
                    trace[-1]['acceptance']='gradient_progress_in_objective_resolution_window'
                    break
            meter['backtracks']+=1;step_size*=.5
        trace[-1]['accepted_step_size']=step_size if accepted else None
        if not accepted:status='line_search_failed';break
    meter['coordinate_row_access_lower_bound']=meter['forward_rows']*d
    return {'weights':w,'status':status,'objective':objective,
      'gradient_norm_diagnostic':gradient_norm,'iterations':meter['accepted_steps'],
      'trace':trace,'meter':meter,'elapsed_seconds':time.perf_counter()-started}


def signed_warm_gradient(old_gradient,weights,lambda_reg,old_count,new_count,
                         removed_features,removed_targets,added_features,added_targets):
    """FP64 audit of the changed-average gradient identity, not a certificate.

    g_new = n_old/n_new*g_old + (sum_added g_i-sum_removed g_i)/n_new
            + (1-n_old/n_new)*lambda*w.
    The count-change regularizer term is essential for average-loss objectives.
    """
    n=_integer(old_count,'old_count',0);m=_integer(new_count,'new_count',0)
    lam=_positive(lambda_reg,'lambda_reg')
    xr,yr,w=_inputs(removed_features,removed_targets,weights)
    xa,ya,_=_inputs(added_features,added_targets,weights)
    g=np.asarray(old_gradient)
    if g.dtype!=np.float64 or g.shape!=w.shape or not np.isfinite(g).all():raise ValueError('Finite FP64 old gradient required')
    if m!=n-len(xr)+len(xa):raise ValueError('Selection counts and signed rows disagree')
    if not m:return lam*w
    r=xr.astype(np.float64);a=xa.astype(np.float64)
    delta=a.T@(expit(a@w)-ya)-r.T@(expit(r@w)-yr)
    return (n/m)*g+delta/m+(1-n/m)*lam*w


def _outward(lo,hi,bits):
    """Exact floor/ceiling to a fixed dyadic grid; never nearest rounding."""
    scale=1<<bits
    return (Fraction((lo.numerator*scale)//lo.denominator,scale),
      Fraction(-((-hi.numerator*scale)//hi.denominator),scale))


def sigmoid_interval(value,*,bits=128,max_terms=256,max_squarings=64):
    """Rigorous rational enclosure; see CONVEX_README.txt for the tail proof.

    No platform exp/log/sigmoid value is trusted by this verification routine.
    """
    bits=_integer(bits,'bits');terms_cap=_integer(max_terms,'max_terms')
    squaring_cap=_integer(max_squarings,'max_squarings',0)
    if bits>4096:raise CertificateBudgetError('Requested dyadic precision exceeds hard verification cap')
    t=rational(value)
    if not t:return Fraction(1,2),Fraction(1,2),{'taylor_terms':0,'squarings':0}
    u=abs(t);s=0
    while u>Fraction(1,2):
        u/=2;s+=1
        if s>squaring_cap:raise CertificateBudgetError('Logit range reduction exceeds configured squaring budget')
    # S_K <= exp(u) <= S_K + a_(K+1)/(1-u/(K+2)).
    total=Fraction(1);term=Fraction(1);target=Fraction(1,1<<(bits+16))
    for k in range(terms_cap):
        nxt=term*u/(k+1)
        tail=nxt/(1-u/Fraction(k+2))
        if tail<=target:break
        total+=nxt;term=nxt
    else:raise CertificateBudgetError('Taylor enclosure did not meet precision within configured term budget')
    lo,hi=_outward(1/(total+tail),1/total,bits+16)
    # Both endpoints are nonnegative. Squaring is monotone; each step is enclosed.
    for _ in range(s):lo,hi=_outward(lo*lo,hi*hi,bits+16)
    if t>0:a,b=1/(1+hi),1/(1+lo)
    else:a,b=lo/(1+lo),hi/(1+hi)
    a,b=_outward(a,b,bits)
    if not (0<=a<=b<=1):raise AssertionError('Internal sigmoid enclosure invariant failed')
    return a,b,{'taylor_terms':k+1,'squarings':s}


def _fraction_json(value):return {'numerator':str(value.numerator),'denominator':str(value.denominator)}
def _hash_array(value):
    a=np.ascontiguousarray(value);return hashlib.sha256(str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes()).hexdigest()


def certify_logistic(features,targets,lambda_reg,weights,*,bits=128,max_terms=256,
                     max_squarings=64,max_coordinates=2_000_000,parameter_tolerance=Fraction(1,10**8)):
    """Exact stored-value optimization certificate, with capped verification work.

    If g_j lies in [L_j,U_j], B²=sum max(|L_j|,|U_j|)² bounds ||g||².
    lambda-strong convexity gives ||w-w*||² <= B²/lambda² and objective gap
    <=B²/(2lambda). Every quantity in these inequalities is exact rational.
    """
    started=time.perf_counter();x,y,w=_inputs(features,targets,weights)
    lam_float=_positive(lambda_reg,'lambda_reg');lam=rational(lam_float)
    cap=_integer(max_coordinates,'max_coordinates',0);n,d=x.shape
    bits=_integer(bits,'bits');max_terms=_integer(max_terms,'max_terms');max_squarings=_integer(max_squarings,'max_squarings',0)
    if bits>4096:raise CertificateBudgetError('Requested dyadic precision exceeds hard verification cap')
    if n*d>cap:raise CertificateBudgetError('Exact gradient row-coordinate budget exceeded before evaluation')
    tolerance=rational(parameter_tolerance)
    if tolerance<0:raise ValueError('Parameter tolerance must be nonnegative')
    wr=[rational(v) for v in w];lower=[lam*v for v in wr];upper=lower.copy()
    counts={'exact_logit_row_evaluations':0,'exact_feature_coordinates':0,'taylor_terms':0,'squarings':0}
    for row,target in zip(x,y):
        xr=[rational(v) for v in row];yr=rational(target)
        t=sum((a*b for a,b in zip(xr,wr)),Fraction(0))
        a,b,cost=sigmoid_interval(t,bits=bits,max_terms=max_terms,max_squarings=max_squarings)
        counts['exact_logit_row_evaluations']+=1;counts['exact_feature_coordinates']+=d
        counts['taylor_terms']+=cost['taylor_terms'];counts['squarings']+=cost['squarings']
        for j,value in enumerate(xr):
            left=value*(a-yr)/n;right=value*(b-yr)/n
            lower[j]+=min(left,right);upper[j]+=max(left,right)
    gradient_squared=sum((max(abs(a),abs(b))**2 for a,b in zip(lower,upper)),Fraction(0))
    parameter_squared=gradient_squared/(lam*lam)
    radius=sqrt_upper_rational(parameter_squared,30)
    gap=gradient_squared/(2*lam)
    result={'schema':'ccu-fractional-logistic-certificate-v1','status':'certified_stored_value_optimization_bound',
      'objective':'single-output mean fractional BCE + lambda/2 * squared Euclidean weight norm; no intercept',
      'empty_target':'zero data loss; unique minimizer zero',
      'records':n,'dimension':d,'lambda_exact':_fraction_json(lam),'sigmoid_interval_precision_bits':bits,
      'gradient_lower':[_fraction_json(v) for v in lower],'gradient_upper':[_fraction_json(v) for v in upper],
      'gradient_norm_squared_upper':_fraction_json(gradient_squared),
      'parameter_error_squared_upper':_fraction_json(parameter_squared),
      'parameter_error_norm_upper':_fraction_json(radius),'objective_gap_upper':_fraction_json(gap),
      'parameter_tolerance':_fraction_json(tolerance),'meets_parameter_tolerance':parameter_squared<=tolerance*tolerance,
      'radius_float_for_display_only':float(radius),'work':counts,'elapsed_seconds':time.perf_counter()-started,
      'input_bindings':{'features_sha256':_hash_array(x),'targets_sha256':_hash_array(y),'weights_sha256':_hash_array(w),
                       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
      'statistical_unlearning_certificate':False,'privacy_claim':False,'canonical_state_claim':False,
      'peak_memory_measured':False,'resource_limits':{'max_coordinates':cap,'max_terms':max_terms,'max_squarings':max_squarings}}
    encoded=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result['sha256']=hashlib.sha256(encoded).hexdigest()
    return result
