#!/usr/bin/env python3
"""Independent algebraic audit, not synthetic empirical observations or labels."""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase3 import calibration


def pmf(N, n, K, j):
    if not max(0, n - N + K) <= j <= min(n, K):
        return Fraction(0)
    return Fraction(math.comb(K,j)*math.comb(N-K,n-j), math.comb(N,n))


def brute_lower(N,n,x,alpha):
    for K in range(N+1):
        if sum((pmf(N,n,K,j) for j in range(x,n+1)), Fraction(0)) > alpha:
            return K
    raise AssertionError('No accepted population')


def run():
    alpha=Fraction(1,20)
    cells=coverage_cells=0
    max_noncoverage=Fraction(0)
    equality_boundary_cases=0
    for N in range(1,25):
        for n in range(1,N+1):
            bounds={}
            for x in range(n+1):
                expected=brute_lower(N,n,x,alpha)
                actual=calibration.finite_population_lower_bound(N,n,x,alpha)
                assert actual['defined'] is True
                assert actual['lower_success_count']==expected, (N,n,x,expected,actual)
                frac=actual['lower_proportion']
                assert Fraction(frac['numerator'],frac['denominator'])==Fraction(expected,N)
                bounds[x]=expected
                cells += 1
            for K in range(N+1):
                noncoverage=sum((pmf(N,n,K,x) for x in range(n+1) if bounds[x]>K), Fraction(0))
                assert noncoverage <= alpha, (N,n,K,noncoverage)
                max_noncoverage=max(max_noncoverage,noncoverage)
                equality_boundary_cases += int(noncoverage==alpha)
                coverage_cells += 1
    extra=[]
    for N,n,x in [(1000,200,0),(1000,200,200),(1000000,200,190),(200,200,181),(200,200,180),(17,0,0),(0,0,0)]:
        result=calibration.finite_population_lower_bound(N,n,x)
        if n:
            L=result['lower_success_count']
            tail=sum((pmf(N,n,L,j) for j in range(x,n+1)),Fraction(0))
            assert tail>alpha
            if L:
                previous=sum((pmf(N,n,L-1,j) for j in range(x,n+1)),Fraction(0))
                assert previous<=alpha
        else:
            assert result['defined'] is False and result['lower_proportion'] is None
        extra.append({'N':N,'n':n,'x':x,'result':result})
    rejected=[]
    for args in [(-1,0,0),(1,2,1),(2,1,2),(True,1,0),(2,True,0),(2,1,False),(2.0,1,0)]:
        try: calibration.finite_population_lower_bound(*args)
        except ValueError: rejected.append(list(args))
        else: raise AssertionError(('Invalid accepted',args))
    for a in [Fraction(0), Fraction(1), True, float('nan')]:
        try: calibration.finite_population_lower_bound(2,1,0,a)
        except ValueError: pass
        else: raise AssertionError(('Invalid alpha accepted',a))
    return {'audit':'independent exhaustive exact finite-population inversion and coverage',
            'evidence_role':'algebraic software tests; no empirical labels or synthetic corpus',
            'passed':True,'alpha':str(alpha),'exhaustive_population_max':24,
            'bound_cells':cells,'coverage_cells':coverage_cells,
            'maximum_exact_noncoverage':str(max_noncoverage),
            'coverage_exact_alpha_cells':equality_boundary_cases,
            'large_and_empty_cases':extra,'invalid_count_cases_rejected':len(rejected),
            'invalid_alpha_cases_rejected':4,
            'production_sha256':hashlib.sha256(Path(calibration.__file__).read_bytes()).hexdigest()}

if __name__=='__main__':
    result=run()
    out=Path(__file__).with_name('independent_finite_population_audit.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'large_and_empty_cases'}},indent=2))
