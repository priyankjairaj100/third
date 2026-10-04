"""Exact, byte-canonical indexed aggregate repair on rational input moments.

FP32 features and FP64 labels are interpreted as exact dyadic rationals. Each
coefficient retains only its unique pivot-identity chart of range([G,H]),
its pivot Gram/cross moments and an integer count. Identity rows are implicit.
No prior chart, original rows, labels, graph or deletion history persists.
Canonical JSON bytes are a logical serialization contract, not allocator erasure.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
from typing import Iterable
import numpy as np
from .summary import _metadata, _integer

ZERO=F(0)
ONE=F(1)
_NATIVE=Path(__file__).with_name('native')/'exact_chart'

def _transpose(a,cols):
    return [[a[i][j] for i in range(len(a))] for j in range(cols)]

def _mul(a,b,c):
    out=[[ZERO for _ in range(c)] for _ in a]
    for i,row in enumerate(a):
        for k,x in enumerate(row):
            if x:
                for j,y in enumerate(b[k]):
                    if y:out[i][j]+=x*y
    return out

def _rref(a,cols):
    a=[list(row) for row in a];piv=[];r=0
    for j in range(cols):
        p=next((i for i in range(r,len(a)) if a[i][j]),None)
        if p is None:continue
        a[r],a[p]=a[p],a[r];v=a[r][j]
        a[r]=[x/v for x in a[r]]
        for i in range(len(a)):
            if i!=r and a[i][j]:
                v=a[i][j];a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        piv.append(j);r+=1
        if r==len(a):break
    return a[:r],piv

def _canonical_python(w,a,b,d,c):
    t=len(a); rows,piv=_rref(_transpose(w,t),d);u=_transpose(rows,d);r=len(piv)
    coord=[w[p] for p in piv]
    aa=_mul(_mul(coord,a,t),_transpose(coord,t),r);bb=_mul(coord,b,c)
    lower,lp=_rref(_transpose([aa[i]+bb[i] for i in range(r)],r+c),r)
    s=len(lp)
    if s<r:
        z=_mul(u,_transpose(lower,r),s);rows,piv=_rref(_transpose(z,s),d)
        rr=[u[p] for p in piv];aa=_mul(_mul(rr,aa,r),_transpose(rr,r),s);bb=_mul(rr,bb,c)
        u=_transpose(rows,d)
    return u,aa,bb,piv

def _canonical(w,a,b,d,c,count,backend='auto'):
    if backend not in ('auto','native','python'):raise ValueError('unknown exact backend')
    native=backend!='python' and _NATIVE.is_file()
    if backend=='native' and not native:raise RuntimeError('compile native/exact_chart.cpp first')
    if not native:return _canonical_python(w,a,b,d,c)
    header=f'{d} {len(a)} {c} {count}\n'
    payload=header+'\n'.join(' '.join(str(x) for x in row) for matrix in (w,a,b) for row in matrix)+'\n'
    run=subprocess.run([str(_NATIVE)],input=payload,text=True,capture_output=True,check=True)
    it=iter(run.stdout.split());dd,r,cc,nn=[int(next(it)) for _ in range(4)]
    if (dd,cc,nn)!=(d,c,count):raise RuntimeError('invalid native dimensions')
    piv=[int(next(it)) for _ in range(r)]
    read=lambda n,m:[[F(next(it)) for _ in range(m)] for _ in range(n)]
    u,aa,bb=read(d,r),read(r,r),read(r,c)
    if next(it,None) is not None:raise RuntimeError('trailing native data')
    return u,aa,bb,piv

def _qjson(q):return [str(q.numerator),str(q.denominator)]
def _qunjson(v):
    if not isinstance(v,list) or len(v)!=2:raise ValueError('invalid rational')
    q=F(int(v[0]),int(v[1]))
    if _qjson(q)!=v:raise ValueError('noncanonical rational encoding')
    return q

@dataclass(frozen=True,slots=True)
class ExactFactor:
    d:int
    c:int
    count:int
    pivots:tuple
    nonpivot:tuple
    gram_upper:tuple
    cross:tuple

    @property
    def rank(self):return len(self.pivots)
    def factor(self):
        r=self.rank;pmap={p:i for i,p in enumerate(self.pivots)};it=iter(self.nonpivot)
        u=[[ONE if pmap[i]==j else ZERO for j in range(r)] if i in pmap else list(next(it)) for i in range(self.d)]
        a=[[ZERO for _ in range(r)] for _ in range(r)];k=0
        for i in range(r):
            for j in range(i,r):a[i][j]=a[j][i]=self.gram_upper[k];k+=1
        return u,a,[list(row) for row in self.cross],self.count
    def zero(self):return self.count==0 and self.rank==0
    @classmethod
    def from_core(cls,w,a,b,d,c,count,backend='auto'):
        d,c,count=_integer(d,'dimension'),_integer(c,'response_dimension'),_integer(count,'count')
        if d<1 or c<1 or not -(2**63)<=count<2**63:raise ValueError('invalid dimensions or native count range')
        t=len(a)
        if len(w)!=d or len(b)!=t or any(len(row)!=t for row in w) or any(len(row)!=t for row in a) or any(len(row)!=c for row in b):raise ValueError('inconsistent exact factor dimensions')
        w=[[F(x) for x in row] for row in w];a=[[F(x) for x in row] for row in a];b=[[F(x) for x in row] for row in b]
        if any(a[i][j]!=a[j][i] for i in range(t) for j in range(i)):raise ValueError('Gram core must be exactly symmetric')
        u,aa,bb,piv=_canonical(w,a,b,d,c,count,backend)
        pset=set(piv);r=len(piv)
        return cls(d,c,int(count),tuple(piv),tuple(tuple(row) for i,row in enumerate(u) if i not in pset),
                   tuple(aa[i][j] for i in range(r) for j in range(i,r)),tuple(tuple(row) for row in bb))
    def add(self,other,backend='auto'):
        if (self.d,self.c)!=(other.d,other.c):raise ValueError('factor dimensions differ')
        u,a,b,n=self.factor();v,aa,bb,nn=other.factor();r,s=self.rank,other.rank
        w=[x+y for x,y in zip(u,v)]
        core=[row+[ZERO]*s for row in a]+[[ZERO]*r+row for row in aa]
        return self.from_core(w,core,b+bb,self.d,self.c,n+nn,backend)
    def moments(self):
        u,a,b,n=self.factor();return _mul(_mul(u,a,self.rank),_transpose(u,self.rank),self.d),_mul(u,b,self.c),n
    def to_json(self):
        return {'count':self.count,'pivots':list(self.pivots),'nonpivot':[[_qjson(q) for q in row] for row in self.nonpivot],
                'gram_upper':[_qjson(q) for q in self.gram_upper],'cross':[[_qjson(q) for q in row] for row in self.cross]}
    @classmethod
    def from_json(cls,obj,d,c):
        if type(obj['count']) is not int or any(type(p) is not int for p in obj['pivots']):raise ValueError('integer count/pivots required')
        return cls(d,c,obj['count'],tuple(obj['pivots']),tuple(tuple(_qunjson(q) for q in row) for row in obj['nonpivot']),
                   tuple(_qunjson(q) for q in obj['gram_upper']),tuple(tuple(_qunjson(q) for q in row) for row in obj['cross']))

class ExactCanonicalSummary:
    __slots__=('universe_size','horizon','dimension','response_dimension','alive','_coeff','_inverse','_degrees')
    @classmethod
    def build(cls,features,targets,blockers,horizon,owners=None,universe_size=None,alive=None,backend='auto'):
        x=np.asarray(features,dtype=np.float32);y=np.asarray(targets,dtype=np.float64)
        if y.ndim==1:y=y[:,None]
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)!=len(blockers) or min(x.shape[1],y.shape[1])<1:raise ValueError('invalid dimensions')
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('finite inputs required')
        universe,h,records,keys,_=_metadata(blockers,owners,horizon,universe_size)
        state=cls.__new__(cls);state.universe_size=universe;state.horizon=h;state.dimension=x.shape[1];state.response_dimension=y.shape[1]
        state.alive=set(range(universe)) if alive is None else {_integer(u,'alive identifier') for u in alive}
        if any(u<0 or u>=universe for u in state.alive) or h>len(state.alive):raise ValueError('invalid alive universe/horizon')
        activeowners=range(len(x)) if owners is None else owners
        if not set(activeowners)<=state.alive:raise ValueError('all record owners must be alive')
        grouped={key:[] for key in keys}
        for row,tail,head in records:
            grouped[tail].append((row,1))
            if head is not None:grouped[head].append((row,-1))
        state._coeff={};d,c=state.dimension,state.response_dimension
        for key,vals in sorted(grouped.items()):
            value=None
            # A constant key can contain the entire corpus. Bound construction
            # cores by d instead of materializing an N-by-N diagonal matrix.
            block_size=min(d,128)
            for start in range(0,len(vals),block_size):
                part=vals[start:start+block_size];t=len(part)
                w=[[F(float(x[i,j])) for i,s in part] for j in range(d)]
                a=[[F(part[i][1]) if i==j else ZERO for j in range(t)] for i in range(t)]
                b=[[s*F(float(v)) for v in y[i]] for i,s in part]
                chunk=ExactFactor.from_core(w,a,b,d,c,sum(s for _,s in part),backend)
                value=chunk if value is None else value.add(chunk,backend)
            if not value.zero():state._coeff[key]=value
        state._reindex();return state
    def _reindex(self):
        self._inverse={};self._degrees={}
        for key in self._coeff:
            self._degrees.setdefault(len(key),set()).add(key)
            for u in key:self._inverse.setdefault(u,set()).add(key)
    def delete(self,identifiers:Iterable[int],backend='auto'):
        req=[_integer(u,'request identifier') for u in identifiers]
        if len(set(req))!=len(req) or not set(req)<=self.alive or len(req)>self.horizon:raise ValueError('duplicate, invalid, retried or over-budget request')
        if not req:return {'deletions':0,'coefficient_collisions':0,'exact_cancellations':0}
        # Stage complete current-state result before commit; arithmetic failures
        # leave the live object untouched. No history survives a successful call.
        removed=set(req);newh=self.horizon-len(req);out={};collisions=cancel=0
        for key,value in sorted(self._coeff.items()):
            newkey=tuple(u for u in key if u not in removed)
            if len(newkey)>newh:continue
            if newkey in out:
                value=out[newkey].add(value,backend);collisions+=1
            if value.zero():out.pop(newkey,None);cancel+=1
            else:out[newkey]=value
        self._coeff=out;self.alive-=removed;self.horizon=newh;self._reindex()
        return {'deletions':len(req),'coefficient_collisions':collisions,'exact_cancellations':cancel}
    def coefficient_keys(self):return tuple(sorted(self._coeff))
    def factor(self,key=()):
        value=self._coeff.get(tuple(key))
        return value.factor() if value is not None else ([[] for _ in range(self.dimension)],[],[],0)
    getfactor=factor
    def exact_moments(self,key=()):
        value=self._coeff.get(tuple(key))
        if value is None:return [[ZERO]*self.dimension for _ in range(self.dimension)],[[ZERO]*self.response_dimension for _ in range(self.dimension)],0
        return value.moments()
    def canonical_bytes(self):
        data={'format':'exact-canonical-ridge-v1','universe_size':self.universe_size,'horizon':self.horizon,'dimension':self.dimension,
              'response_dimension':self.response_dimension,'alive':sorted(self.alive),
              'coefficients':[[list(k),self._coeff[k].to_json()] for k in sorted(self._coeff)]}
        return (json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n').encode('ascii')
    def save(self,path):
        data=self.canonical_bytes();Path(path).write_bytes(data);return len(data)
    @classmethod
    def load(cls,path,validate=True):
        return cls.from_bytes(Path(path).read_bytes(),validate=validate)
    @classmethod
    def from_bytes(cls,raw,validate=True):
        """Load canonical bytes; validate=False skips costly rank recomputation.

        Shape, type, membership and unique-encoding checks remain mandatory.
        The fast option is intended only for an already trusted current state.
        """
        obj=json.loads(raw);s=cls.__new__(cls)
        if obj['format']!='exact-canonical-ridge-v1':raise ValueError('unsupported exact format')
        for name in ('universe_size','horizon','dimension','response_dimension'):
            if type(obj[name]) is not int:raise ValueError('integer metadata required')
            setattr(s,name,obj[name])
        if any(type(u) is not int for u in obj['alive']):raise ValueError('integer public identifiers required')
        s.alive=set(obj['alive']);s._coeff={}
        for k,v in obj['coefficients']:
            if any(type(u) is not int for u in k):raise ValueError('integer key identifiers required')
            key=tuple(k)
            if key in s._coeff:raise ValueError('duplicate coefficient')
            s._coeff[key]=ExactFactor.from_json(v,s.dimension,s.response_dimension)
        s._reindex();s.check_invariants(recanonicalize=validate)
        if s.canonical_bytes()!=raw:raise ValueError('noncanonical state serialization')
        return s
    def check_invariants(self,recanonicalize=False):
        def require(condition,message):
            if not condition:raise ValueError(message)
        require(all(type(getattr(self,k)) is int for k in ('universe_size','horizon','dimension','response_dimension')),'integer metadata required')
        require(self.dimension>0 and self.response_dimension>0 and self.universe_size>=0 and 0<=self.horizon<=len(self.alive),'invalid dimensions/horizon')
        require(all(type(u) is int and 0<=u<self.universe_size for u in self.alive),'invalid public universe membership')
        for key,v in self._coeff.items():
            r=v.rank
            require(all(type(u) is int for u in key) and key==tuple(sorted(set(key))) and set(key)<=self.alive and len(key)<=self.horizon and not v.zero(),'invalid coefficient key')
            require(type(v.count) is int and -(2**63)<=v.count<2**63,'invalid coefficient count')
            require((v.d,v.c)==(self.dimension,self.response_dimension),'coefficient dimension mismatch')
            require(v.pivots==tuple(sorted(set(v.pivots))) and all(type(p) is int and 0<=p<v.d for p in v.pivots),'invalid pivot rows')
            require(len(v.nonpivot)==v.d-r and all(len(row)==r for row in v.nonpivot),'invalid basis shape')
            require(len(v.gram_upper)==r*(r+1)//2 and len(v.cross)==r and all(len(row)==v.c for row in v.cross),'invalid core shape')
            if recanonicalize:
                u,a,b,n=v.factor();require(ExactFactor.from_core(u,a,b,v.d,v.c,n)==v,'coefficient is not in canonical chart')
        oldi,oldd=self._inverse,self._degrees;self._reindex();require(oldi==self._inverse and oldd==self._degrees,'derived index mismatch')
    def decode_candidate(self,lambda_reg):
        lam=float(lambda_reg)
        if not math.isfinite(lam) or lam<=0:raise ValueError('positive finite regularizer required')
        u,a,b,n=self.factor()
        if n<0:raise ValueError('negative target count')
        if not n or not a:return np.zeros((self.dimension,self.response_dimension))
        uu=np.asarray(u,dtype=np.float64);aa=np.asarray(a,dtype=np.float64);bb=np.asarray(b,dtype=np.float64)
        q,r=np.linalg.qr(uu,mode='reduced');g=r@aa@r.T;h=r@bb
        g=(g+g.T)*0.5;g.flat[::len(g)+1]+=lam*n
        return q@np.linalg.solve(g,h)
    def accounting(self):
        vals=list(self._coeff.values());fractions=[q for v in vals for row in v.nonpivot for q in row]+[q for v in vals for q in v.gram_upper]+[q for v in vals for row in v.cross for q in row]
        bitbytes=sum((abs(q.numerator).bit_length()+7)//8+(q.denominator.bit_length()+7)//8 for q in fractions)
        seen=set()
        def size(o):
            if id(o) in seen:return 0
            seen.add(id(o));n=sys.getsizeof(o)
            if isinstance(o,dict):n+=sum(size(k)+size(v) for k,v in o.items())
            elif isinstance(o,(tuple,list,set)):n+=sum(size(v) for v in o)
            elif isinstance(o,F):n+=size(o.numerator)+size(o.denominator)
            elif isinstance(o,ExactFactor):n+=sum(size(getattr(o,x)) for x in o.__slots__)
            return n
        return {'method':'exact-pivot-chart-rational','coefficient_count':len(vals),'rank_sum':sum(v.rank for v in vals),'maximum_rank':max((v.rank for v in vals),default=0),
                'stored_rational_scalars':len(fractions),'rational_integer_payload_bytes':bitbytes,'canonical_serialized_bytes':len(self.canonical_bytes()),
                'maximum_numerator_bits':max((abs(q.numerator).bit_length() for q in fractions),default=0),'maximum_denominator_bits':max((q.denominator.bit_length() for q in fractions),default=0),
                'python_live_bytes_estimate':sys.getsizeof(self)+sum(size(getattr(self,x)) for x in self.__slots__),
                'retained_feature_rows':0,'retained_target_rows':0,'retained_original_graph':False,'remaining_horizon':self.horizon,
                'guarantee':'Exact rational coefficient semantics and byte-canonical logical serialization; arbitrary precision cost explicit.'}
