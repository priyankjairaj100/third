"""Independent native-chart rational checks, using dense ambient RREF oracle."""
from fractions import Fraction as F
import json
from pathlib import Path
import random
import subprocess
from audit_certificates_independent import mm, trans


def rref(matrix, width):
    a = [row[:] for row in matrix]
    rank, pivots = 0, []
    for col in range(width):
        pivot = next((i for i in range(rank,len(a)) if a[i][col]),None)
        if pivot is None:
            continue
        a[rank],a[pivot] = a[pivot],a[rank]
        scale = a[rank][col]
        a[rank] = [value/scale for value in a[rank]]
        for i in range(len(a)):
            if i != rank:
                scale = a[i][col]
                a[i] = [a[i][j]-scale*a[rank][j] for j in range(width)]
        pivots.append(col)
        rank += 1
        if rank == len(a):
            break
    return a[:rank],pivots


def native(w,a,b,count,c):
    d,t = len(w),len(a)
    words = [str(d),str(t),str(c),str(count)]
    words += [str(x) for matrix in (w,a,b) for row in matrix for x in row]
    binary = Path(__file__).parent/'ccu/native/exact_chart'
    p = subprocess.run([str(binary)],input=' '.join(words),text=True,capture_output=True,check=True)
    words = iter(p.stdout.split())
    dd,r,cc,n = [int(next(words)) for _ in range(4)]
    piv = [int(next(words)) for _ in range(r)]
    u = [[F(next(words)) for _ in range(r)] for _ in range(dd)]
    aa = [[F(next(words)) for _ in range(r)] for _ in range(r)]
    bb = [[F(next(words)) for _ in range(cc)] for _ in range(r)]
    assert list(words) == []
    assert dd==d and cc==c and n==count
    return u,aa,bb,piv


def run():
    rng = random.Random(8206)
    checks = 0
    for d in range(1,7):
        for t in range(0,d+2):
            for c in (1,3):
                vals = lambda n,m:[[F(rng.randrange(-2,3),rng.randrange(1,5))
                                    for _ in range(m)] for _ in range(n)]
                w,v,b = vals(d,t),vals(t,t),vals(t,c)
                a = [[v[i][j]+v[j][i] for j in range(t)] for i in range(t)]
                # Include joint-rank cancellation and H-only coefficients.
                if t % 3 == 0:
                    a = [[F(0) for _ in range(t)] for _ in range(t)]
                if t % 5 == 0:
                    b = [[F(0) for _ in range(c)] for _ in range(t)]
                gram = mm(mm(w,a,t),trans(w,t),d)
                cross = mm(w,b,c)
                full = [gram[i]+cross[i] for i in range(d)]
                rows,piv = rref(trans(full,d+c),d)
                expected_u = trans(rows,d)
                expected_a = [[gram[i][j] for j in piv] for i in piv]
                expected_b = [cross[i] for i in piv]
                u,aa,bb,pp = native(w,a,b,-3,c)
                assert (u,aa,bb,pp) == (expected_u,expected_a,expected_b,piv)
                assert mm(mm(u,aa,len(aa)),trans(u,len(aa)),d) == gram
                assert mm(u,bb,c) == cross
                # Idempotent canonical encoding of already canonical factors.
                assert native(u,aa,bb,-3,c) == (u,aa,bb,pp)
                checks += 1
    large_checks = 0
    for bits in (127,521,1024):
        large = 2**bits
        w = [[F(large+1,large-1),F(3,7)], [F(1,11),F(large-3,large+3)],
             [F(large+9,19),F(2,large+7)]]
        a = [[F(large+5,large-9),F(7,13)],[F(7,13),F(17,large+23)]]
        b = [[F(large-1,29)], [F(31,large+1)]]
        gram = mm(mm(w,a,2),trans(w,2),3)
        cross = mm(w,b,1)
        rows,piv = rref(trans([gram[i]+cross[i] for i in range(3)],4),3)
        expected = (trans(rows,3),[[gram[i][j] for j in piv] for i in piv],
                    [cross[i] for i in piv],piv)
        assert native(w,a,b,2,1) == expected
        large_checks += 1
    return {'independent_ambient_rref_comparisons':checks,
            'reconstruction_checks':checks,'canonical_idempotence_checks':checks,
            'large_rational_bitlength_checks':large_checks,
            'large_input_numerator_bitlengths':[128,522,1025],
            'failed_checks':0,'scope':'Exact algebra fixtures; not empirical data experiments.'}


if __name__=='__main__':
    result = run()
    out = Path(__file__).parent/'canonical_results'
    out.mkdir(exist_ok=True)
    (out/'independent_native_chart_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
