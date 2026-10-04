"""Audit exact certificates; arithmetic fixtures are not empirical datasets."""
from fractions import Fraction as F
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ccu.certified_ridge import (CertificateInputError, exact_residual_certificate,
                                  factor_residual_certificate, rational,
                                  sqrt_upper_rational)
from ccu.data import lexical_engineering_features, read_natural_jsonl


def matmul(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))),F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(z) for z in zip(*a)]


def exact_solve(a,b):
    """Independent rational Gauss-Jordan solve for small audit matrices."""
    n,c=len(a),len(b[0]); aug=[list(a[i])+list(b[i]) for i in range(n)]
    for k in range(n):
        p=next(i for i in range(k,n) if aug[i][k])
        aug[k],aug[p]=aug[p],aug[k]
        pivot=aug[k][k]; aug[k]=[v/pivot for v in aug[k]]
        for i in range(n):
            if i!=k:
                mult=aug[i][k]
                aug[i]=[aug[i][j]-mult*aug[k][j] for j in range(n+c)]
    return [row[n:] for row in aug]


def main():
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    def rejects(name,fun):
        try:fun()
        except CertificateInputError:check(name,True)
        else:raise AssertionError(name)

    g=[[F(2),F(1)],[F(1),F(2)]]; h=[[F(1)],[F(0)]]
    exact=[[F(3,8)],[F(-1,8)]]
    cert=exact_residual_certificate(g,h,2,F(1,2),exact)
    check('known_exact_solution_zero_residual',cert.error_squared_upper_bound==0)
    candidate=[[.4],[-.1]]
    perturbed=exact_residual_certificate(g,h,2,F(1,2),candidate)
    true_error=sum((rational(candidate[i][0])-exact[i][0])**2 for i in range(2))
    check('perturbed_solution_bound_encloses_independent_exact_error',true_error<=perturbed.error_squared_upper_bound)
    check('candidate_digest_binds_weights',perturbed.candidate_digest!=cert.candidate_digest)
    check('same_target_digest_independent_candidate',perturbed.target_digest==cert.target_digest)

    u=[[F(2),F(0)],[F(1),F(1)],[F(0),F(3)]]
    a=[[F(3),F(-1)],[F(-1),F(1)]]; b=[[F(2),F(1)],[F(-1),F(4)]]
    gg=matmul(matmul(u,a),transpose(u)); hh=matmul(u,b)
    lam=F(1,100); n=5
    mm=[[gg[i][j]+(lam*n if i==j else 0) for j in range(3)]for i in range(3)]
    ww=exact_solve(mm,hh); rounded=[[float(v) for v in row]for row in ww]
    fc=factor_residual_certificate(u,a,b,n,lam,rounded)
    dc=exact_residual_certificate(gg,hh,n,lam,rounded)
    check('nonorthogonal_basis_factor_dense_exact_residual_agreement',fc.residual_squared_frobenius==dc.residual_squared_frobenius)
    error=sum((rational(rounded[i][j])-ww[i][j])**2 for i in range(3) for j in range(2))
    check('nonorthogonal_basis_bound_contains_exact_gauss_jordan_solution',error<=fc.error_squared_upper_bound)
    check('exact_psd_rank',fc.psd_core_rank==2 and dc.psd_core_rank==2)
    sing=exact_residual_certificate([[1,1],[1,1]],[[0],[0]],1,1,[[1],[2]])
    check('singular_psd_accepted',sing.psd_core_rank==1)
    zero=factor_residual_certificate([[],[]],[],[],4,1,[[1],[2]])
    check('zero_rank_factor_nonempty_error',zero.error_squared_upper_bound==5)
    empty=factor_residual_certificate([[],[]],[],[],0,1,[[1],[2]])
    check('empty_target_nonzero_candidate_exact_error',empty.error_squared_upper_bound==5 and empty.residual_squared_frobenius==0)
    check('empty_target_zero_candidate_exact_bound',exact_residual_certificate([[0]],[[0]],0,1,[[0]]).error_squared_upper_bound==0)
    rejects('indefinite_zero_pivot_rejected',lambda:exact_residual_certificate([[0,1],[1,0]],[[0],[0]],1,1,[[0],[0]]))
    rejects('negative_gram_rejected',lambda:exact_residual_certificate([[-1]],[[0]],1,1,[[0]]))
    rejects('nonsymmetric_gram_rejected',lambda:exact_residual_certificate([[1,1],[0,1]],[[0],[0]],1,1,[[0],[0]]))
    rejects('indefinite_factor_core_rejected',lambda:factor_residual_certificate([[1]], [[-1]], [[0]],1,1,[[0]]))
    rejects('empty_nonzero_moments_rejected',lambda:exact_residual_certificate([[1]],[[0]],0,1,[[0]]))
    rejects('empty_factor_nonzero_gram_rejected',lambda:factor_residual_certificate([[1]],[[1]],[[0]],0,1,[[0]]))
    rejects('empty_factor_nonzero_cross_rejected',lambda:factor_residual_certificate([[1]],[[0]],[[1]],0,1,[[0]]))
    rejects('nan_candidate_rejected',lambda:exact_residual_certificate([[1]],[[0]],1,1,[[float('nan')]]))
    rejects('zero_regularization_rejected',lambda:exact_residual_certificate([[1]],[[0]],1,0,[[0]]))
    rejects('noninteger_count_rejected',lambda:exact_residual_certificate([[1]],[[0]],1.1,1,[[0]]))
    rejects('negative_count_rejected',lambda:exact_residual_certificate([[1]],[[0]],-1,1,[[0]]))
    rejects('shape_mismatch_rejected',lambda:exact_residual_certificate([[1,0]],[[0]],1,1,[[0]]))
    rejects('extended_numpy_float_rejected',lambda:rational(np.longdouble('0.1')))
    check('numpy_float32_preserved_exactly',rational(np.float32('0.1'))==F.from_float(float(np.float32('0.1'))))
    check('numpy_float64_preserved_exactly',rational(np.float64('0.1'))==F.from_float(float(np.float64('0.1'))))
    for i,square in enumerate([F(0),F(1,9),F(2),F(1,10**100),F(10**100+1,7)]):
        for p in [0,8,30,80]:
            up=sqrt_upper_rational(square,p)
            check(f'integer_sqrt_bound_{i}_{p}',up*up>=square and (up==0 or (up-F(1,10**p))**2<square))
    check('exact_tolerance_comparison',fc.meets_tolerance(fc.upper_bound()) and not perturbed.meets_tolerance(0))
    check('prediction_bound',perturbed.prediction_squared_upper_bound([3,4])==25*perturbed.error_squared_upper_bound)

    # Real natural text, first 12 unchanged Civil preview rows, fixed lexical map.
    records=read_natural_jsonl(Path(__file__).parent/'data/civil_comments_engineering_preview.jsonl')[:12]
    features=lexical_engineering_features(records,dimension=16)
    if isinstance(features,tuple):features=features[0]
    x=np.asarray(features,dtype=np.float64)
    y=np.asarray([row['label'] for row in records],dtype=np.float64)[:,None]
    exact_x=[[rational(v) for v in row]for row in x]
    exact_y=[[rational(v) for v in row]for row in y]
    basis=transpose(exact_x)
    identity=[[F(i==j) for j in range(len(records))]for i in range(len(records))]
    gram=matmul(basis,exact_x); cross=matmul(basis,exact_y)
    lam_float=0.01
    floating=(x.T@x)+(lam_float*len(records))*np.eye(x.shape[1])
    candidate=cho_solve(cho_factor(floating,lower=True),x.T@y)
    t=time.perf_counter()
    real=factor_residual_certificate(basis,identity,exact_y,len(records),lam_float,candidate)
    factor_seconds=time.perf_counter()-t
    t=time.perf_counter()
    dense=exact_residual_certificate(gram,cross,len(records),lam_float,candidate)
    dense_seconds=time.perf_counter()-t
    check('natural_text_factor_dense_certificate_agreement',real.error_squared_upper_bound==dense.error_squared_upper_bound)
    m=[[gram[i][j]+(rational(lam_float)*len(records) if i==j else 0) for j in range(x.shape[1])] for i in range(x.shape[1])]
    true=exact_solve(m,cross)
    true_error=sum((rational(candidate[i,0])-true[i][0])**2 for i in range(x.shape[1]))
    check('natural_text_certificate_contains_independent_exact_solution',true_error<=real.error_squared_upper_bound)
    report={'schema':'certificate-audit-v1','checks_passed':len(checks),'failures':0,'checks':checks,
            'arithmetic_fixtures_are_not_empirical_datasets':True,
            'real_text_check':{'records':12,'dataset':'Civil Comments engineering preview','feature_dimension':16,
                'representation':'Unmodified real text with fixed lexical engineering features; no semantic/E5 claim',
                'factor_certificate_seconds':factor_seconds,'dense_certificate_seconds':dense_seconds,
                'certificate':real.to_dict(40),'independent_exact_solution_error_squared':{'numerator':str(true_error.numerator),'denominator':str(true_error.denominator)}}}
    out=Path(__file__).parent/'results/certified_ridge_audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(checks),'failures':0,'real_error_upper':real.to_dict(40)['error_upper_bound_decimal'],'factor_certificate_seconds':factor_seconds,'dense_certificate_seconds':dense_seconds}))


if __name__=='__main__':main()
