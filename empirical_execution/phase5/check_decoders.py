"""Independent rational/software and natural-preview decoder checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fractions import Fraction as Q
import hashlib,json,math
import numpy as np
from ccu.core import RidgeMoments,ridge_moments
from ccu.data import read_natural_jsonl,lexical_engineering_features
from phase5.decoders import decode,extended_audit,DEFAULT_POLICY

def rational_solve(a,b):
    n=len(a);c=len(b[0]);table=[list(a[i])+list(b[i]) for i in range(n)]
    for j in range(n):
        pivot=next(i for i in range(j,n) if table[i][j]);table[j],table[pivot]=table[pivot],table[j]
        v=table[j][j];table[j]=[x/v for x in table[j]]
        for i in range(n):
            if i!=j:
                v=table[i][j];table[i]=[x-v*y for x,y in zip(table[i],table[j])]
    return [r[n:] for r in table]

def run():
    checks=[];cases=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    # These tiny exact integer/dyadic matrices are algebraic software fixtures,
    # not synthetic empirical data. The second output is a genuinely zero RHS.
    x=np.array([[1,0,1],[0,2,1],[1,1,0],[-1,0,2]],np.float32)
    y=np.array([[.25,0],[.5,0],[.75,0],[1,0]],np.float64);m=ridge_moments(x,y);lam=.125
    a=[[sum(Q(float(row[i]))*Q(float(row[j])) for row in x)+(Q(lam)*len(x) if i==j else 0) for j in range(3)] for i in range(3)]
    h=[[sum(Q(float(row[i]))*Q(float(target[j])) for row,target in zip(x,y)) for j in range(2)] for i in range(3)]
    exact=rational_solve(a,h)
    for solver in ('cholesky','cg'):
        r=decode(m,lam,solver=solver);check(solver+'_release',r.diagnostics['release_allowed']);w=r.weights
        rr=[[sum(a[i][k]*Q(float(w[k,j])) for k in range(3))-h[i][j] for j in range(2)] for i in range(3)]
        residual2=sum(v*v for row in rr for v in row);error2=sum((Q(float(w[i,j]))-exact[i][j])**2 for i in range(3) for j in range(2))
        check(solver+'_independent_exact_error_bounded',error2<=residual2/(Q(lam)*len(x))**2)
        check(solver+'_close_exact_solution',float(error2)<1e-16)
        check(solver+'_zero_rhs_exact',np.array_equal(w[:,1],np.zeros(3)))
        check(solver+'_normalized_residual_gate',r.diagnostics['normalized_residual_eta']<=1e-10)
        if solver=='cg':check('cg_zero_rhs_no_iterations',r.diagnostics['iteration_counts'][1]==0)
        audit=extended_audit(x,y,w,lam,computed_moments=m)
        check(solver+'_extended_not_certified',not audit['numerically_certified'] and not audit['service_failure_overridden'])
        cases.append({'role':'algebraic_software_fixture','solver':r.diagnostics,'extended_audit':audit,
                      'exact_error_squared':[str(error2.numerator),str(error2.denominator)],'exact_residual_squared':[str(residual2.numerator),str(residual2.denominator)]})
    empty=RidgeMoments(np.zeros((3,3)),np.zeros((3,2)),0)
    for solver in ('cholesky','cg'):
        r=decode(empty,.01,solver=solver);check(solver+'_empty_zero',r.diagnostics['release_allowed'] and not np.any(r.weights))
        bad=RidgeMoments(np.eye(3)*1e-300,np.zeros((3,2)),0);r=decode(bad,.01,solver=solver)
        check(solver+'_empty_drift_rejected',not r.diagnostics['release_allowed'] and r.diagnostics['status']=='empty_count_nonzero_moments')
    asymmetric=m.gram.copy();asymmetric[0,1]+=1e-15
    check('asymmetry_rejected',decode(RidgeMoments(asymmetric,m.cross,m.count),lam).diagnostics['status']=='invalid_asymmetric_stored_gram')
    bad=decode(RidgeMoments(-np.eye(3),m.cross,m.count),lam)
    check('nonpositive_system_rejected',not bad.diagnostics['release_allowed'] and bad.diagnostics['audit_requested'])
    under=decode(RidgeMoments(np.eye(1),np.array([[1e-320]]),1),.01,solver='cg')
    check('cg_underflow_not_zero_rhs',under.diagnostics['status']=='cg_squared_norm_underflow' and not under.diagnostics['release_allowed'])
    altered={**DEFAULT_POLICY,'normalized_residual_limit':1e-5}
    try:decode(m,lam,policy=altered)
    except ValueError:check('policy_cannot_relax_after_results',True)
    else:check('policy_cannot_relax_after_results',False)
    ill=RidgeMoments(np.diag([0.,1.,1e8]),np.array([[1.],[1.],[1.]]),1)
    for solver in ('cholesky','cg'):
        r=decode(ill,1e-8,solver=solver);check(solver+'_ill_condition_reported',r.diagnostics['condition_number_2_estimate']>=1e15)
        check(solver+'_ill_condition_no_fallback',not r.diagnostics['fallback_performed'])
        cases.append({'role':'conditioning_software_fixture','solver':r.diagnostics})
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)
    xx,_=lexical_engineering_features(rows,64);yy=np.asarray([r['label'] for r in rows],np.float64)[:,None]
    for retained in (0,30,len(rows)):
        moment=ridge_moments(xx[:retained],yy[:retained])
        for solver in ('cholesky','cg'):
            r=decode(moment,.01,solver=solver);check(f'natural_{retained}_{solver}_release',r.diagnostics['release_allowed'])
            audit=extended_audit(xx[:retained],yy[:retained],r.weights,.01,computed_moments=moment)
            check(f'natural_{retained}_{solver}_extended_count',audit['computed_count_equal'])
            if retained:
                independently=np.linalg.solve(moment.gram+.01*retained*np.eye(64),moment.cross)
                discrepancy=float(np.linalg.norm(r.weights-independently))
                check(f'natural_{retained}_{solver}_agreement',discrepancy<1e-6)
            else:discrepancy=0.
            cases.append({'role':'natural_Civil100_lexical_decoder_software_check_only','retained_prefix':retained,'solver':r.diagnostics,'extended_audit':audit,'independent_solve_discrepancy_fro':discrepancy})
    report={'schema':'ccu-decoder-checks-1','passed':True,'checks':checks,'check_count':len(checks),'cases':cases,
            'semantic_or_source_empirical_evidence':False,'frozen_isolated_worker_run_uses_this_decoder':False,
            'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('decoders.py','check_decoders.py')}}
    out=Path(__file__).parent/'results/decoder_checks.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'checks':len(checks),'path':str(out)}))
if __name__=='__main__':run()
