"""Versioned additions to isolated services. Historical implementations stay fixed."""
from pathlib import Path
import io,json
import numpy as np
from ccu.core import RidgeMoments,ridge_moments
from phase5 import methods as previous
from phase5.payload import PayloadState,_ARRAYS

METHODS=previous.METHODS+('B-F','B-E-FP32')

class FrozenSelection:
    """Wrong-target baseline: train only surviving originally selected rows."""
    method='B-F'
    def __init__(self,graph,x,y,horizon,*,unit='record'):
        if unit not in ('record','source'):raise ValueError('Unknown unit')
        owners=graph.record_ids if unit=='record' else graph.source_ids
        self.unit=unit;self.unit_ids=tuple(sorted(set(owners)));self.alive=set(self.unit_ids)
        if type(horizon)is not int or not 0<=horizon<=len(self.alive):raise ValueError('Invalid horizon')
        self.horizon=horizon;ix=np.flatnonzero(np.diff(graph.indptr)==0)
        self.x=np.asarray(x,dtype=np.float32)[ix].copy();yy=np.asarray(y,dtype=np.float64)
        if yy.ndim==1:yy=yy[:,None]
        self.y=yy[ix].copy();self.owners=tuple(owners[i] for i in ix)
        self.dimension=self.x.shape[1];self.outputs=self.y.shape[1]
        self.copied_payload_bytes=0
    def delete(self,ids):
        ids=list(ids)
        if any(not isinstance(i,str) or i not in self.unit_ids for i in ids):raise ValueError('Unknown unit')
        fresh=set(ids)&self.alive
        if len(fresh)>self.horizon:raise ValueError('Cumulative horizon exhausted')
        self.alive.difference_update(fresh);self.horizon-=len(fresh)
        keep=np.asarray([u not in fresh for u in self.owners],bool)
        removed=int(np.count_nonzero(~keep));copied=0
        if removed:
            self.x=self.x[keep];self.y=self.y[keep];self.owners=tuple(u for u,k in zip(self.owners,keep) if k)
            copied=self.x.nbytes+self.y.nbytes;self.copied_payload_bytes+=copied
        return {'fresh_deletions':len(fresh),'remaining_horizon':self.horizon,'removed_selected_count':removed,
                'admitted_count':0,'payload_copy_output_bytes':copied,'target':'frozen_original_selection_minus_deleted_units'}
    def moments(self):return ridge_moments(self.x,self.y)
    def accounting(self):return {'method':self.method,'unit':self.unit,'target':'frozen_original_selection_minus_deleted_units',
        'payload_bytes':self.x.nbytes+self.y.nbytes,'remaining_horizon':self.horizon,'selected_count':len(self.x),
        'cumulative_payload_copy_output_bytes':self.copied_payload_bytes,'retained_original_graph':False,
        'no_future_admission_claim':True}
    def snapshot_bytes(self):
        m={'schema':'ccu-frozen-selection-1','unit':self.unit,'unit_ids':self.unit_ids,'alive':sorted(self.alive),
           'horizon':self.horizon,'owners':self.owners,'copied_payload_bytes':self.copied_payload_bytes}
        f=io.BytesIO();np.savez(f,metadata=np.asarray(json.dumps(m,sort_keys=True)),x=self.x,y=self.y);return f.getvalue()
    def snapshot(self,path):Path(path).write_bytes(self.snapshot_bytes())
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']));x=z['x'].copy();y=z['y'].copy()
        if m.get('schema')!='ccu-frozen-selection-1':raise ValueError('Invalid snapshot')
        s=cls.__new__(cls);s.unit=m['unit'];s.unit_ids=tuple(m['unit_ids']);s.alive=set(m['alive']);s.horizon=m['horizon'];s.owners=tuple(m['owners'])
        if s.unit not in ('record','source') or len(set(s.unit_ids))!=len(s.unit_ids) or not s.alive<=set(s.unit_ids) or not 0<=s.horizon<=len(s.alive):raise ValueError('Invalid membership')
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(s.owners)!=len(x) or not set(s.owners)<=s.alive or x.dtype!=np.float32 or y.dtype!=np.float64 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid payload')
        s.x=x;s.y=y;s.dimension=x.shape[1];s.outputs=y.shape[1];s.copied_payload_bytes=m['copied_payload_bytes'];return s

class FP32Payload(PayloadState):
    """Named sensitivity. FP32 signed moments; FP64 common decode of stored values."""
    def _signed(self,rows,sign):
        self.gram=self.gram.astype(np.float32,copy=False);self.cross=self.cross.astype(np.float32,copy=False)
        for lo in range(0,len(rows),self.batch_rows):
            block=rows[lo:lo+self.batch_rows];z=self.x[block];yg=self.y[block];y=yg.astype(np.float32)
            g,h=z.T@z,z.T@y
            if sign>0:self.gram+=g;self.cross+=h
            else:self.gram-=g;self.cross-=h
            self.count+=sign*len(block)
            self.cumulative['peak_blas_workspace_array_bytes']=max(self.cumulative['peak_blas_workspace_array_bytes'],z.nbytes+yg.nbytes+y.nbytes+g.nbytes+h.nbytes)
            self.cumulative['blas_batches']+=1;self.cumulative['payload_rows_read']+=len(block)
            self.cumulative['statistic_coordinate_additions']+=g.size+h.size
    def moments(self):return RidgeMoments(self.gram.astype(np.float64),self.cross.astype(np.float64),self.count)
    def accounting(self):return {**super().accounting(),'method':'B-A-FP32' if self.retain_all else 'B-E-FP32','moment_precision':'FP32','decoder_precision':'FP64',
        'target_values_in_update_cast_to_FP32':True,'approximation_frontier_not_FP64_exactness':True}
    def snapshot_bytes(self):
        raw=super().snapshot_bytes()
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        m=json.loads(arrays['metadata'].tobytes());m['schema']='ccu-packed-payload-FP32-1'
        arrays['metadata']=np.frombuffer(json.dumps(m,sort_keys=True,separators=(',',':')).encode(),np.uint8)
        f=io.BytesIO();np.savez(f,**arrays);return f.getvalue()
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={*_ARRAYS,'metadata'}:raise ValueError('Invalid members')
            m=json.loads(z['metadata'].tobytes());s=cls.__new__(cls)
            if m.pop('schema',None)!='ccu-packed-payload-FP32-1':raise ValueError('Invalid FP32 schema')
            for k,v in m.items():setattr(s,k,v)
            for k in _ARRAYS:setattr(s,k,z[k].copy())
        if s.gram.dtype!=np.float32 or s.cross.dtype!=np.float32:raise ValueError('FP32 moments required')
        s.unit_ids=tuple(s.unit_ids);s.record_ids=tuple(s.record_ids);s._unit_lookup={r:i for i,r in enumerate(s.unit_ids)}
        # Reuse structural checks with exact promotion; no relaxed moment equality gate.
        g,h=s.gram,s.cross;s.gram=g.astype(np.float64);s.cross=h.astype(np.float64)
        try:s.check_invariants(check_moments=False)
        finally:s.gram,s.cross=g,h
        return s

def build_method(method,graph,features,targets,horizon,**kwargs):
    if method=='B-F':return FrozenSelection(graph,features,targets,horizon,unit=kwargs.get('unit','record'))
    if method=='B-E-FP32':return FP32Payload(graph,features,targets,horizon,unit=kwargs.get('unit','record'),
        compaction_fraction=kwargs.get('compaction_fraction',.5),batch_rows=kwargs.get('batch_rows',1024))
    return previous.build_method(method,graph,features,targets,horizon,**kwargs)

def load_method(method,path):
    if method=='B-F':return FrozenSelection.load(path)
    if method=='B-E-FP32':return FP32Payload.load(path)
    return previous.load_method(method,path)

# FP32 is a state/update sensitivity. Every variant uses the same FP64 decoder.
# Counts remain exact through an explicit n < 2**24 construction guard.
PRIMARY_METHODS=('O-G','O-T','B-E','B-A','P-I','P-S','P-R','B-F')
FP32_METHODS=tuple(m+'-FP32' for m in PRIMARY_METHODS)
METHODS=previous.METHODS+('B-F',)+FP32_METHODS

def _tag_snapshot(raw,tag,*,restore=False,promote_values=False):
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    m=json.loads(str(arrays['metadata']))
    if restore:
        if m.pop('precision_variant',None)!=tag:raise ValueError('Wrong precision variant')
    else:m['precision_variant']=tag
    arrays['metadata']=np.asarray(json.dumps(m,sort_keys=True))
    if promote_values:arrays['values']=arrays['values'].astype(np.float64)
    f=io.BytesIO();np.savez(f,**arrays);return f.getvalue()

class FP32Summary(previous.SummaryService):
    def __init__(self,graph,features,targets,horizon,*,unit='record',method='P-I'):
        from ccu.summary import IndexedRidgeSummary,_metadata,_Entry
        if len(features)>=2**24:raise ValueError('FP32 exact count guard requires n < 2**24')
        self.method,self.unit=method,unit
        self.unit_ids=tuple(graph.record_ids) if unit=='record' else tuple(sorted(set(graph.source_ids)))
        lookup={s:i for i,s in enumerate(self.unit_ids)};owners=None if unit=='record' else [lookup[s] for s in graph.source_ids]
        universe,h,records,keys,_=_metadata(graph.blockers,owners,horizon,len(self.unit_ids))
        x=np.asarray(features,np.float32);y=np.asarray(targets,np.float32)
        if y.ndim==1:y=y[:,None]
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid FP32 payload')
        d,c=x.shape[1],y.shape[1];q=d*(d+1)//2;p=q+d*c+1;tri=np.triu_indices(d)
        st=IndexedRidgeSummary.__new__(IndexedRidgeSummary)
        st.universe_size=universe;st.horizon=h;st.dimension=d;st.response_dimension=c;st.packed_dimension=p
        st.alive=set(range(universe));st._coeff={};st._inverse={};st._degrees={}
        for v,tail,head in records:
            stat=np.empty(p,np.float32);stat[:q]=np.outer(x[v],x[v])[tri];stat[q:-1]=np.outer(x[v],y[v]).ravel();stat[-1]=1.
            for key,sign in ((tail,1),(head,-1)):
                if key is None:continue
                if key not in st._coeff:st._coeff[key]=_Entry(key,stat.copy() if sign==1 else -stat)
                elif sign==1:st._coeff[key].value+=stat
                else:st._coeff[key].value-=stat
        for key in list(st._coeff):
            ent=st._coeff[key]
            if not np.any(ent.value!=0):del st._coeff[key];continue
            st._degrees.setdefault(len(key),set()).add(ent)
            for u in key:st._inverse.setdefault(u,set()).add(ent)
        self.state=st;self.dimension=d;self.response_dimension=c;self.packed_dimension=p
        if method!='P-I':
            self.horizon=h;self.alive=st.alive;coeff=st.coefficients()
            if method=='P-S':self.coeff=coeff
            elif method=='P-R':self.groups=sorted(previous._groups(records,keys),key=lambda v:v[0]);self._compress(coeff)
            else:raise ValueError('Unknown FP32 summary')
            del self.state
    def _compress(self,coeff):
        self.coords={}
        for nodes,ground in self.groups:
            for key in (nodes if ground else nodes[1:]):self.coords[key]=coeff.get(key,np.zeros(self.packed_dimension,np.float32)).astype(np.float32,copy=True)
    def coefficients(self):
        if self.method!='P-R':return super().coefficients()
        out={k:v.copy() for k,v in self.coords.items()}
        for nodes,ground in self.groups:
            if not ground:
                v=np.zeros(self.packed_dimension,np.float32)
                for k in nodes[1:]:v-=out[k]
                out[nodes[0]]=v
        return out
    def moments(self):
        m=super().moments();return RidgeMoments(m.gram.astype(np.float64),m.cross.astype(np.float64),m.count)
    def accounting(self):return {**super().accounting(),'method':self.method+'-FP32','coefficient_precision':'FP32',
        'coefficient_build_update_reconstruction_precision':'FP32','decoder_precision':'FP64','exact_count_guard':'n < 2**24',
        'numeric_guarantee':'approximation frontier; no FP64 moment equality claim'}
    def snapshot_bytes(self):
        raw=previous.SummaryService.snapshot(self,None)
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        a['values']=a['values'].astype(np.float32)  # Includes the empty coefficient matrix.
        f=io.BytesIO();np.savez(f,**a)
        return _tag_snapshot(f.getvalue(),'summary-FP32-1')
    def snapshot(self,path):Path(path).write_bytes(self.snapshot_bytes())
    @classmethod
    def load(cls,path):
        raw=Path(path).read_bytes()
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            if z['values'].dtype!=np.float32:raise ValueError('FP32 coefficient state required')
        # The old structural validator requires FP64. Its temporary exact
        # promotion is charged to load cost, then discarded before restriction.
        s=previous.SummaryService.load(io.BytesIO(_tag_snapshot(raw,'summary-FP32-1',restore=True,promote_values=True)))
        s.__class__=cls
        if s.method=='P-I':
            for e in s.state._coeff.values():e.value=e.value.astype(np.float32)
        else:
            vals=s.coeff if s.method=='P-S' else s.coords
            for k in vals:vals[k]=vals[k].astype(np.float32)
        return s

class FP32Oracle(previous.OracleService):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if len(self.x)>=2**24:raise ValueError('FP32 exact count guard requires n < 2**24')
        self.y=self.y.astype(np.float32)
    def moments(self):
        ix=self.selected_indices();z=self.x[ix];y=self.y[ix]
        return RidgeMoments((z.T@z).astype(np.float64),(z.T@y).astype(np.float64),len(ix))
    def accounting(self):return {**super().accounting(),'method':self.method+'-FP32','retraining_moments_precision':'FP32',
        'targets_stored_FP32':True,'curator_scoring_precision':'unchanged_FP64_fixed_graph_contract','decoder_precision':'FP64'}
    def snapshot_bytes(self):return _tag_snapshot(previous.OracleService.snapshot(self,None),'oracle-FP32-1')
    def snapshot(self,path):Path(path).write_bytes(self.snapshot_bytes())
    @classmethod
    def load(cls,path):
        raw=_tag_snapshot(Path(path).read_bytes(),'oracle-FP32-1',restore=True)
        s=previous.OracleService.load(io.BytesIO(raw));s.__class__=cls
        if s.x.dtype!=np.float32 or s.y.dtype!=np.float32:raise ValueError('FP32 oracle payload required')
        return s

class FP32Frozen(FrozenSelection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if len(self.x)>=2**24:raise ValueError('FP32 exact count guard requires n < 2**24')
        self.y=self.y.astype(np.float32)
    def moments(self):return RidgeMoments((self.x.T@self.x).astype(np.float64),(self.x.T@self.y).astype(np.float64),len(self.x))
    def accounting(self):return {**super().accounting(),'method':'B-F-FP32','retraining_moments_precision':'FP32','targets_stored_FP32':True,'decoder_precision':'FP64'}
    def snapshot_bytes(self):return _tag_snapshot(super().snapshot_bytes(),'frozen-FP32-1')
    @classmethod
    def load(cls,path):
        raw=_tag_snapshot(Path(path).read_bytes(),'frozen-FP32-1',restore=True)
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if arrays['y'].dtype!=np.float32:raise ValueError('FP32 target state required')
        arrays['y']=arrays['y'].astype(np.float64);f=io.BytesIO();np.savez(f,**arrays);f.seek(0)
        s=FrozenSelection.load(f);s.__class__=cls;s.y=s.y.astype(np.float32);return s

_previous_build=build_method
_previous_load=load_method

def build_method(method,graph,features,targets,horizon,**kwargs):
    if method.endswith('-FP32'):
        base=method[:-5];unit=kwargs.get('unit','record')
        if len(features)>=2**24:raise ValueError('FP32 exact count guard requires n < 2**24')
        if base in ('P-I','P-S','P-R'):return FP32Summary(graph,features,targets,horizon,unit=unit,method=base)
        if base in ('O-G','O-T'):return FP32Oracle(graph,features,targets,horizon,unit=unit,method=base,curator=kwargs.get('curator'))
        if base=='B-F':return FP32Frozen(graph,features,targets,horizon,unit=unit)
        if base in ('B-E','B-A'):return FP32Payload(graph,features,targets,horizon,unit=unit,retain_all=base=='B-A',compaction_fraction=kwargs.get('compaction_fraction',.5),batch_rows=kwargs.get('batch_rows',1024))
        raise ValueError('Unknown FP32 variant')
    return _previous_build(method,graph,features,targets,horizon,**kwargs)

def load_method(method,path):
    if method.endswith('-FP32'):
        base=method[:-5]
        if base in ('P-I','P-S','P-R'):return FP32Summary.load(path)
        if base in ('O-G','O-T'):return FP32Oracle.load(path)
        if base=='B-F':return FP32Frozen.load(path)
        if base in ('B-E','B-A'):return FP32Payload.load(path)
        raise ValueError('Unknown FP32 variant')
    return _previous_load(method,path)

class CachedGraphOracle(previous.OracleService):
    """Named O-G sensitivity with the same retained CSR kernel as O-T."""
    def __init__(self,graph,x,y,horizon,*,unit='record'):
        super().__init__(graph,x,y,horizon,unit=unit,method='O-T')
    def accounting(self):return {**super().accounting(),'method':'O-G-cached',
        'tier':'cached_exhaustive_graph_plus_all_learner_payload','kernel_equivalent_to':'O-T',
        'graph_construction_charged_in_shared_bundle':True,'per_release_scalar_graph_recompute':False}
    def snapshot_bytes(self):
        raw=previous.OracleService.snapshot(self,None)
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        m=json.loads(str(a['metadata']));m['comparison_tier']='O-G-cached';a['metadata']=np.asarray(json.dumps(m,sort_keys=True))
        f=io.BytesIO();np.savez(f,**a);return f.getvalue()
    def snapshot(self,path):Path(path).write_bytes(self.snapshot_bytes())
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if json.loads(str(z['metadata'])).get('comparison_tier')!='O-G-cached':raise ValueError('Cached tier required')
        s=previous.OracleService.load(path);s.__class__=cls;return s

_full_build=build_method
_full_load=load_method
METHODS=METHODS+('O-G-cached',)

def build_method(method,graph,features,targets,horizon,**kwargs):
    if method=='O-G-cached':return CachedGraphOracle(graph,features,targets,horizon,unit=kwargs.get('unit','record'))
    return _full_build(method,graph,features,targets,horizon,**kwargs)

def load_method(method,path):
    if method=='O-G-cached':return CachedGraphOracle.load(path)
    state=_full_load(method,path)
    if method.endswith('-FP32') and state.accounting()['method']!=method:raise ValueError('Snapshot method/precision mismatch')
    return state
