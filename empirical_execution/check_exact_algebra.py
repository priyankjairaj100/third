"""Algebra/software fixtures only; not synthetic empirical data."""
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from ccu.exact_canonical import ExactFactor,_mul,_transpose

def main():
    checks=[]
    cases=[
        ([[1],[2],[0]],[[0]],[[3]],0), # Gram cancellation with surviving H
        ([[1,1],[2,2],[0,0]],[[1,0],[0,-1]],[[2],[1]],0),
        ([[1,0],[0,1],[1,1]],[[0,1],[1,0]],[[1,2],[3,4]],-2),
        ([[0],[0],[0]],[[1]],[[1]],3),
    ]
    for d in (2,3,5,8):
        for t in (1,2,3,5):
            w=[[F(((i+2)*(j+1)+j*j)%7-3,2**((i+j)%3)) for j in range(t)] for i in range(d)]
            a=[[F((i+j+1)%5-2,2**((i+j)%2)) for j in range(t)] for i in range(t)]
            b=[[F((i+2*j)%4-1,2**(i%3)) for j in range(2)] for i in range(t)]
            cases.append((w,a,b,t))
    for num,(w,a,b,n) in enumerate(cases):
        d=len(w);c=len(b[0]);t=len(a)
        w=[[F(x) for x in row] for row in w];a=[[F(x) for x in row] for row in a];b=[[F(x) for x in row] for row in b]
        native=ExactFactor.from_core(w,a,b,d,c,n,'native');python=ExactFactor.from_core(w,a,b,d,c,n,'python')
        assert native==python
        g,h,count=native.moments()
        assert g==_mul(_mul(w,a,t),_transpose(w,t),d) and h==_mul(w,b,c) and count==n
        u,aa,bb,_=native.factor();again=ExactFactor.from_core(u,aa,bb,d,c,n,'native');assert again==native
        doubled=native.add(native);gg,hh,nn=doubled.moments()
        assert gg==[[2*x for x in row] for row in g] and hh==[[2*x for x in row] for row in h] and nn==2*n
        checks.append({'case':num,'dimension':d,'rank':native.rank,'native_python_equal':True,'exact_moments_equal':True,'idempotent':True,'merge':True})
    result={'scope':'exact algebra software fixtures, no empirical dataset','cases':len(checks),'checks':checks,'failures':0}
    out=Path(__file__).with_name('canonical_results');out.mkdir(exist_ok=True);(out/'algebra_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(checks),'failures':0}))
if __name__=='__main__':main()
