"""Independent algebraic checks; these are mathematical fixtures, not data experiments."""
from fractions import Fraction as F
import json
from pathlib import Path
import random
from ccu.certified_ridge import (exact_residual_certificate,
    factor_residual_certificate, sqrt_upper_rational, CertificateInputError)


def mm(a, b, width):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))), F(0))
             for j in range(width)] for i in range(len(a))]


def trans(a, width):
    return [[a[i][j] for i in range(len(a))] for j in range(width)]


def solve(a, b):
    # Independent exact Gauss-Jordan solve, with pivoting.
    d, c = len(a), len(b[0])
    ab = [list(a[i]) + list(b[i]) for i in range(d)]
    for k in range(d):
        pivot = next(i for i in range(k, d) if ab[i][k])
        ab[k], ab[pivot] = ab[pivot], ab[k]
        scale = ab[k][k]
        ab[k] = [x/scale for x in ab[k]]
        for i in range(d):
            if i != k:
                scale = ab[i][k]
                ab[i] = [ab[i][j]-scale*ab[k][j] for j in range(d+c)]
    return [row[d:] for row in ab]


def run():
    rng = random.Random(1401)
    checks = 0
    for d in range(1, 7):
        for r in range(0, d+1):
            for c in (1, 2, 3):
                for trial in range(3):
                    vals = lambda n, m: [[F(rng.randrange(-3,4),rng.randrange(1,5))
                                         for _ in range(m)] for _ in range(n)]
                    u, v, b = vals(d,r), vals(r,r), vals(r,c)
                    a = mm(v,trans(v,r),r)
                    gram = mm(mm(u,a,r),trans(u,r),d)
                    cross = mm(u,b,c)
                    count, lam = 1+trial, F(1+trial, 7)
                    head = vals(d,c)
                    hessian = [[gram[i][j]+(lam*count if i==j else 0)
                                for j in range(d)] for i in range(d)]
                    oracle = solve(hessian,cross)
                    err = sum(((head[i][j]-oracle[i][j])**2
                               for i in range(d) for j in range(c)), F(0))
                    dense = exact_residual_certificate(gram,cross,count,lam,head)
                    factor = factor_residual_certificate(u,a,b,count,lam,head)
                    assert dense.residual_squared_frobenius == factor.residual_squared_frobenius
                    assert dense.error_squared_upper_bound == factor.error_squared_upper_bound
                    assert err <= factor.error_squared_upper_bound
                    assert factor.upper_bound()**2 >= err
                    checks += 1
    sqrt_checks = 0
    for numerator in range(0,31):
        for denominator in range(1,17):
            for places in (0,1,4,30):
                squared = F(numerator,denominator)
                upper = sqrt_upper_rational(squared,places)
                assert upper**2 >= squared
                below = upper-F(1,10**places)
                assert below < 0 or below**2 < squared
                sqrt_checks += 1
    rejected = 0
    for gram in ([[F(-1)]], [[F(0),F(1)],[F(1),F(0)]],
                 [[F(1),F(2)],[F(2),F(1)]], [[F(1),F(1)],[F(0),F(1)]]):
        d = len(gram)
        try:
            exact_residual_certificate(gram,[[F(0)] for _ in range(d)],1,F(1),
                                       [[F(0)] for _ in range(d)])
        except CertificateInputError:
            rejected += 1
        else:
            raise AssertionError('Indefinite/asymmetric target accepted')
    empty = factor_residual_certificate([[F(1),F(0)],[F(0),F(0)]],
             [[F(0),F(0)],[F(0),F(1)]], [[F(0)],[F(1)]], 0,F(1),[[F(2)],[F(3)]])
    assert empty.error_squared_upper_bound == 13
    empty_dense = exact_residual_certificate([[F(0)]],[[F(0)]],0,F(1),[[F(7,3)]])
    assert empty_dense.error_squared_upper_bound == F(49,9)
    return {'independent_exact_solves':checks,'integer_sqrt_checks':sqrt_checks,
            'invalid_psd_rejections':rejected,'empty_target_checks':2,
            'failed_checks':0,
            'scope':'Algebraic verification fixtures, not empirical datasets or a formal proof assistant.'}


if __name__ == '__main__':
    result = run()
    folder = Path(__file__).parent/'canonical_results'
    folder.mkdir(exist_ok=True)
    (folder/'independent_certificate_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
