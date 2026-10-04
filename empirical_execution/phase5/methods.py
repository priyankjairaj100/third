"""Versioned service states: dense indexed, dense scan, incidence-rank scan.

P-R is structural coefficient-rank compression, not feature/joint-span compression.
Its omitted coefficient is minus the sum of the other coefficients in an
ungrounded incidence component. Metadata stores components, not record edges.
Floating sums remain diagnostics; canonical here means the basis selection/order.
"""
from pathlib import Path
import io,json,sys,zipfile
import numpy as np
from ccu.summary import IndexedRidgeSummary, _metadata
from ccu.core import RidgeMoments

METHODS=('B-E','B-A','P-I','P-S','P-R','P-I-jointspan','B-E-compact','O-G','O-T')

def _groups(records,keys):
    parent={k:k for k in keys};parent[None]=None
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    for _,a,b in records:
        a,b=root(a),root(b)
        if a!=b:parent[a]=b
    out={}
    for k in keys:out.setdefault(root(k),[]).append(k)
    return [(tuple(sorted(v)),root(None)==r) for r,v in out.items()]

def _mapped_groups(groups,deleted,horizon):
    """Exact image of component incidence subspaces under quotient/truncation."""
    mapped=[];keys=set()
    for nodes,grounded in groups:
        dest=set();ground=grounded
        for key in nodes:
            new=tuple(u for u in key if u not in deleted)
            if len(new)>horizon:ground=True
            else:dest.add(new)
        keys.update(dest);mapped.append((dest,ground))
    parent={k:k for k in keys};parent[None]=None
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    for nodes,ground in mapped:
        seq=sorted(nodes)+([None] if ground else [])
        for a,b in zip(seq,seq[1:]):
            a,b=root(a),root(b)
            if a!=b:parent[a]=b
    out={}
    for k in keys:out.setdefault(root(k),[]).append(k)
    return sorted([(tuple(sorted(v)),root(None)==r) for r,v in out.items()],key=lambda x:x[0])

def _unpack(value,d,c):
    q=d*(d+1)//2; tri=np.triu_indices(d);gram=np.zeros((d,d));gram[tri]=value[:q];gram[(tri[1],tri[0])]=value[:q]
    n=float(value[-1])
    if not np.isfinite(value).all() or n<0 or not n.is_integer():raise FloatingPointError('Invalid reconstructed moments/count')
    return RidgeMoments(gram,value[q:-1].reshape(d,c).copy(),int(n))

class SummaryService:
    def __init__(self,graph,features,targets,horizon,*,unit='record',method='P-I'):
        if method not in ('P-I','P-S','P-R') or unit not in ('record','source'):raise ValueError('Unknown service')
        self.method,self.unit=method,unit
        self.unit_ids=tuple(graph.record_ids) if unit=='record' else tuple(sorted(set(graph.source_ids)))
        lookup={s:i for i,s in enumerate(self.unit_ids)}
        owners=None if unit=='record' else [lookup[s] for s in graph.source_ids]
        self.state=IndexedRidgeSummary.build(features,targets,graph.blockers,horizon,owners=owners,universe_size=len(self.unit_ids))
        self.dimension=self.state.dimension;self.response_dimension=self.state.response_dimension
        self.packed_dimension=self.state.packed_dimension
        if method!='P-I':
            self.horizon=self.state.horizon;self.alive=set(self.state.alive)
            coeff=self.state.coefficients()
            if method=='P-S':self.coeff=coeff
            else:
                _,_,records,keys,_=_metadata(graph.blockers,owners,horizon,len(self.unit_ids))
                self.groups=sorted(_groups(records,keys),key=lambda x:x[0])
                self._compress(coeff)
            del self.state

    @property
    def remaining_horizon(self):return self.state.horizon if self.method=='P-I' else self.horizon

    def _compress(self,coeff):
        self.coords={}
        for nodes,ground in self.groups:
            for key in (nodes if ground else nodes[1:]):
                self.coords[key]=coeff.get(key,np.zeros(self.packed_dimension)).copy()

    def coefficients(self):
        if self.method=='P-I':return self.state.coefficients()
        if self.method=='P-S':return {k:v.copy() for k,v in self.coeff.items()}
        out={k:v.copy() for k,v in self.coords.items()}
        for nodes,ground in self.groups:
            if not ground:
                v=np.zeros(self.packed_dimension)
                for k in nodes[1:]:v-=out[k]
                out[nodes[0]]=v
        return out

    def delete(self,identifiers):
        ids=list(identifiers)
        if any(not isinstance(i,str) for i in ids):raise ValueError('String IDs required')
        lookup={s:i for i,s in enumerate(self.unit_ids)}
        if any(i not in lookup for i in ids):raise ValueError('Unknown unit ID')
        alive=self.state.alive if self.method=='P-I' else self.alive
        fresh=sorted({lookup[i] for i in ids}&alive)
        if len(fresh)>self.remaining_horizon:raise ValueError('Cumulative horizon exhausted')
        if self.method=='P-I':meter=self.state.delete(fresh)
        else:
            coeff=self.coefficients();dead=set(fresh);h=self.horizon-len(fresh);out={}
            meter={'scanned_keys':len(coeff),'coefficient_collisions':0,'horizon_pruned_keys':0,'reconstructed_coordinate_vectors':len(coeff)}
            for key in sorted(coeff):
                new=tuple(u for u in key if u not in dead)
                if len(new)>h:meter['horizon_pruned_keys']+=1;continue
                if new in out:out[new]+=coeff[key];meter['coefficient_collisions']+=1
                else:out[new]=coeff[key].copy()
            if self.method=='P-S':self.coeff={k:v for k,v in out.items() if np.any(v!=0)}
            else:self.groups=_mapped_groups(self.groups,dead,h);self._compress(out)
            self.alive.difference_update(dead);self.horizon=h
        return {**meter,'requested_units':len(ids),'fresh_deletions':len(fresh),'remaining_horizon':self.remaining_horizon}

    def moments(self):
        if self.method=='P-I':return self.state.moments()
        if self.method=='P-S':value=self.coeff.get((),np.zeros(self.packed_dimension))
        else:value=self.coefficients().get((),np.zeros(self.packed_dimension))
        return _unpack(value,self.dimension,self.response_dimension)

    def accounting(self):
        if self.method=='P-I':out=self.state.accounting()
        else:
            values=self.coeff if self.method=='P-S' else self.coords
            numeric=sum(v.nbytes for v in values.values());keys=list(values)
            out={'numeric_coefficient_bytes':numeric,'stored_coordinate_vectors':len(values),'key_incidences':sum(map(len,keys)),
                 'structural_keys':len(self.coeff) if self.method=='P-S' else sum(len(ns) for ns,g in self.groups),
                 'structural_rank':None if self.method=='P-S' else len(self.coords),
                 'python_container_bytes_lower_bound':sys.getsizeof(values)+sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in values.items()),
                 'remaining_horizon':self.horizon,'retained_feature_rows':0,'retained_target_rows':0,'retained_original_graph':False}
        out.update(method=self.method,unit=self.unit,unit_id_utf8_bytes=sum(len(s.encode()) for s in self.unit_ids),
                   numeric_guarantee='FP64 diagnostic; not exact rational or byte-history canonical')
        return out

    def snapshot(self,path):
        path=None if path is None else Path(path)
        meta={'schema':'ccu-summary-service-1','method':self.method,'unit':self.unit,'unit_ids':self.unit_ids,'dimension':self.dimension,'response_dimension':self.response_dimension,
              'packed_dimension':self.packed_dimension,'horizon':self.remaining_horizon,'alive':sorted(self.state.alive if self.method=='P-I' else self.alive)}
        if self.method=='P-I':coeff=self.state.coefficients()
        elif self.method=='P-S':coeff=self.coeff
        else:coeff=self.coords;meta['groups']=[{'keys':nodes,'grounded':g} for nodes,g in self.groups]
        keys=sorted(coeff);meta['keys']=keys
        values=np.stack([coeff[k] for k in keys]) if keys else np.empty((0,self.packed_dimension))
        if path is None:
            f=io.BytesIO();np.savez(f,metadata=np.asarray(json.dumps(meta)),values=values);return f.getvalue()
        with path.open('wb') as f:np.savez(f,metadata=np.asarray(json.dumps(meta)),values=values)
        return {'serialized_bytes':path.stat().st_size}

    def snapshot_bytes(self):return self.snapshot(None)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['metadata']));values=z['values'].copy()
        if meta.get('schema')!='ccu-summary-service-1' or meta.get('method') not in ('P-I','P-S','P-R') or meta.get('unit') not in ('record','source'):raise ValueError('Invalid snapshot schema/method/unit')
        for k in ('dimension','response_dimension','packed_dimension','horizon'):
            if type(meta[k])is not int or meta[k]<(0 if k=='horizon' else 1):raise ValueError('Invalid snapshot dimension/horizon')
        d,c=meta['dimension'],meta['response_dimension']
        if meta['packed_dimension']!=d*(d+1)//2+d*c+1:raise ValueError('Invalid packed dimension')
        ids=meta['unit_ids'];alive=meta['alive'];keys=meta['keys']
        if any(not isinstance(u,str) or not u for u in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid public IDs')
        if any(type(u)is not int or not 0<=u<len(ids) for u in alive) or alive!=sorted(set(alive)) or meta['horizon']>len(alive):raise ValueError('Invalid alive membership')
        def validkey(key):return isinstance(key,list) and all(type(u)is int and u in alive for u in key) and key==sorted(set(key)) and len(key)<=meta['horizon']
        if not all(validkey(k) for k in keys) or keys!=sorted(keys):raise ValueError('Invalid coefficient keys')
        s=cls.__new__(cls)
        for k in ('method','unit','dimension','response_dimension','packed_dimension','horizon'):setattr(s,k,meta[k])
        s.unit_ids=tuple(meta['unit_ids']);s.alive=set(meta['alive']);coeff={tuple(k):v.copy() for k,v in zip(meta['keys'],values)}
        if len(coeff)!=len(meta['keys']) or values.shape!=(len(coeff),s.packed_dimension) or not np.isfinite(values).all():raise ValueError('Corrupt coefficient snapshot')
        if s.method=='P-I':
            from ccu.summary import _Entry
            st=IndexedRidgeSummary.__new__(IndexedRidgeSummary);st.universe_size=len(s.unit_ids);st.horizon=s.horizon
            st.dimension=s.dimension;st.response_dimension=s.response_dimension;st.packed_dimension=s.packed_dimension;st.alive=s.alive
            st._coeff={};st._inverse={};st._degrees={}
            for key,value in coeff.items():
                ent=_Entry(key,value);st._coeff[key]=ent;st._degrees.setdefault(len(key),set()).add(ent)
                for u in key:st._inverse.setdefault(u,set()).add(ent)
            st.check_invariants();s.state=st
        elif s.method=='P-S':s.coeff=coeff
        elif s.method=='P-R':
            groups=meta['groups'];allkeys=[];coordinatekeys=[]
            for g in groups:
                ns=g['keys']
                if type(g['grounded'])is not bool or not ns or ns!=sorted(ns) or not all(validkey(k) for k in ns):raise ValueError('Invalid rank components')
                allkeys.extend(tuple(k) for k in ns);coordinatekeys.extend(tuple(k) for k in (ns if g['grounded'] else ns[1:]))
            if len(set(allkeys))!=len(allkeys) or set(coordinatekeys)!=set(coeff):raise ValueError('Invalid rank partition/coordinates')
            s.coords=coeff;s.groups=[(tuple(tuple(k) for k in g['keys']),g['grounded']) for g in groups]
        else:raise ValueError('Unknown method snapshot')
        return s

def build_method(method,graph,features,targets,horizon,*,unit='record',compaction_fraction=.5,batch_rows=1024,curator=None):
    if method in ('O-G','O-T'):return OracleService(graph,features,targets,horizon,unit=unit,method=method,curator=curator)
    if method=='P-I-jointspan':return JointService(graph,features,targets,horizon,unit=unit)
    if method=='B-E-compact':return CompactService(graph,features,targets,horizon,unit=unit)
    if method in ('B-E','B-A'):
        from phase5.payload import PayloadState
        return PayloadState(graph,features,targets,horizon,unit=unit,retain_all=method=='B-A',compaction_fraction=compaction_fraction,batch_rows=batch_rows)
    return SummaryService(graph,features,targets,horizon,unit=unit,method=method)

def load_method(method,path):
    if method in ('O-G','O-T'):return OracleService.load(path)
    if method=='P-I-jointspan':return JointService.load(path)
    if method=='B-E-compact':return CompactService.load(path)
    if method in ('B-E','B-A'):
        from phase5.payload import PayloadState
        return PayloadState.load(path)
    return SummaryService.load(path)


class JointService:
    """Explicitly named feature/response joint-span indexed sensitivity."""
    def __init__(self,graph,x,y,horizon,*,unit='record'):
        from ccu.joint_span_summary import JointSpanRidgeSummary
        self.unit=unit;self.unit_ids=tuple(graph.record_ids) if unit=='record' else tuple(sorted(set(graph.source_ids)))
        lookup={s:i for i,s in enumerate(self.unit_ids)}
        owners=None if unit=='record' else [lookup[s] for s in graph.source_ids]
        self.state=JointSpanRidgeSummary.build(x,y,graph.blockers,horizon,owners=owners,universe_size=len(self.unit_ids))
    def delete(self,ids):
        ids=list(ids);lookup={s:i for i,s in enumerate(self.unit_ids)}
        if any(not isinstance(i,str) or i not in lookup for i in ids):raise ValueError('Unknown unit ID')
        fresh=sorted({lookup[i] for i in ids}&self.state.alive)
        if len(fresh)>self.state.horizon:raise ValueError('Cumulative horizon exhausted')
        return self.state.delete(fresh)
    def moments(self):return self.state.moments()
    def accounting(self):return {**self.state.accounting(),'method':'P-I-jointspan','unit':self.unit,'unit_id_utf8_bytes':sum(len(s.encode()) for s in self.unit_ids)}
    def snapshot(self,path):
        path=Path(path);self.state.save(path)
        side=Path(str(path)+'.units.json');side.write_text(json.dumps({'unit':self.unit,'unit_ids':self.unit_ids}))
        return {'serialized_bytes':path.stat().st_size+side.stat().st_size}
    def snapshot_bytes(self):
        # Explicit new in-memory serializer, same frozen NPZ schema; temporary
        # full bytes are charged to repair RSS rather than treated as free.
        s=self.state;keys=sorted(s._coeff);ptr=np.zeros(len(keys)+1,np.int64)
        for i,key in enumerate(keys):ptr[i+1]=ptr[i]+len(key)
        arrays={'format_version':np.array(1,np.int64),'metadata':np.array([s.universe_size,s.horizon,s.dimension,s.response_dimension],np.int64),
                'alive':np.array(sorted(s.alive),np.int64),'indptr':ptr,'indices':np.array([u for k in keys for u in k],np.int64),
                'modes':np.array([s._coeff[k].value.dense is not None for k in keys],np.uint8),'counts':np.array([s._coeff[k].value.count for k in keys],np.int64)}
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            def put(name,value):
                with archive.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(value),allow_pickle=False)
            for name,value in arrays.items():put(name,value)
            for i,key in enumerate(keys):
                v=s._coeff[key].value;prefix=f'coefficient_{i:08d}_'
                for name in (('dense',) if v.dense is not None else ('u','a','b')):put(prefix+name,getattr(v,name))
        return {'state.npz':out.getvalue(),'state.npz.units.json':json.dumps({'unit':self.unit,'unit_ids':self.unit_ids}).encode()}
    @classmethod
    def load(cls,path):
        from ccu.joint_span_summary import JointSpanRidgeSummary
        obj=cls.__new__(cls);meta=json.loads(Path(str(path)+'.units.json').read_text());obj.unit=meta['unit'];obj.unit_ids=tuple(meta['unit_ids'])
        obj.state=JointSpanRidgeSummary.load(path);return obj


class CompactService(JointService):
    """Frozen no-Gram compact eligible baseline with explicit common decoder."""
    def __init__(self,graph,x,y,horizon,*,unit='record'):
        from ccu.compact_payload import CompactEligiblePayloadState
        self.unit=unit;self.unit_ids=tuple(graph.record_ids) if unit=='record' else tuple(sorted(set(graph.source_ids)))
        lookup={s:i for i,s in enumerate(self.unit_ids)};owners=None if unit=='record' else [lookup[s] for s in graph.source_ids]
        self.state=CompactEligiblePayloadState.build(x,y,graph.blockers,horizon,owners=owners,universe_size=len(self.unit_ids))
    def accounting(self):return {**self.state.accounting(),'method':'B-E-compact','unit':self.unit,'unit_id_utf8_bytes':sum(len(s.encode()) for s in self.unit_ids),'persistent_Gram':False,'metadata_backend':'legacy_Python_dictionary_not_CSR'}
    def snapshot(self,path):
        s=self.state;rows=sorted(s._owner)
        meta={'schema':'ccu-compact-service-1','unit':self.unit,'unit_ids':self.unit_ids,'horizon':s.horizon,'alive':sorted(s.alive),
              'dimension':s.dimension,'response_dimension':s.response_dimension,'rows':rows,
              'owners':[s._owner[r] for r in rows],'blockers':[sorted(s._blockers[r]) for r in rows]}
        x=np.stack([s._features[r] for r in rows]) if rows else np.empty((0,s.dimension),np.float32)
        y=np.stack([s._targets[r] for r in rows]) if rows else np.empty((0,s.response_dimension),np.float64)
        if path is None:
            f=io.BytesIO();np.savez(f,metadata=np.asarray(json.dumps(meta)),features=x,targets=y);return f.getvalue()
        with Path(path).open('wb') as f:np.savez(f,metadata=np.asarray(json.dumps(meta)),features=x,targets=y)
        return {'serialized_bytes':Path(path).stat().st_size}
    def snapshot_bytes(self):return self.snapshot(None)
    @classmethod
    def load(cls,path):
        from ccu.compact_payload import CompactEligiblePayloadState
        with np.load(path,allow_pickle=False) as z:m=json.loads(str(z['metadata']));x=z['features'].copy();y=z['targets'].copy()
        if m.get('schema')!='ccu-compact-service-1':raise ValueError('Invalid compact snapshot schema')
        obj=cls.__new__(cls);obj.unit=m['unit'];obj.unit_ids=tuple(m['unit_ids']);s=CompactEligiblePayloadState.__new__(CompactEligiblePayloadState)
        s.universe_size=len(obj.unit_ids);s.horizon=m['horizon'];s.alive=set(m['alive']);s.dimension=m['dimension'];s.response_dimension=m['response_dimension']
        if obj.unit not in ('record','source') or len(set(obj.unit_ids))!=len(obj.unit_ids) or not 0<=s.horizon<=len(s.alive) or not s.alive<=set(range(s.universe_size)):raise ValueError('Invalid compact metadata')
        if x.shape!=(len(m['rows']),s.dimension) or y.shape!=(len(m['rows']),s.response_dimension) or x.dtype!=np.float32 or y.dtype!=np.float64 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid compact arrays')
        s._owner={};s._blockers={};s._features={};s._targets={};s._by_owner={};s._reverse={};s._by_count={};s._selected=set()
        if len(set(m['rows']))!=len(m['rows']) or len(m['owners'])!=len(m['rows']) or len(m['blockers'])!=len(m['rows']):raise ValueError('Invalid compact row metadata')
        for i,r in enumerate(m['rows']):
            own=m['owners'][i];b=set(m['blockers'][i])
            if type(r)is not int or r<0 or own not in s.alive or own in b or not b<=s.alive or len(b)>s.horizon:raise ValueError('Invalid compact eligibility')
            s._owner[r]=own;s._blockers[r]=b;s._features[r]=x[i].copy();s._targets[r]=y[i].copy();s._by_owner.setdefault(own,set()).add(r);s._by_count.setdefault(len(b),set()).add(r)
            for u in b:s._reverse.setdefault(u,set()).add(r)
            if not b:s._selected.add(r)
        obj.state=s;return obj


class OracleService:
    """Greater-payload-access oracle; state retains original payload and membership.

    O-G repeats independent scalar pair scoring on every release. O-T computes
    oracle selected IDs from the fixed CSR and rebuilds moments; graph cost is
    excluded there by definition. Neither is a payload-free maintenance method.
    """
    def __init__(self,graph,x,y,horizon,*,unit='record',method='O-G',curator=None):
        self.method=method;self.unit=unit;self.ids=tuple(graph.record_ids);self.sources=tuple(graph.source_ids)
        self.unit_ids=self.ids if unit=='record' else tuple(sorted(set(self.sources)))
        self.alive=set(self.unit_ids);self.horizon=horizon
        if type(horizon)is not int or not 0<=horizon<=len(self.alive):raise ValueError('Invalid oracle horizon')
        self.x=np.asarray(x,np.float32).copy();self.y=np.asarray(y,np.float64).copy()
        if self.y.ndim==1:self.y=self.y[:,None]
        self.priority=graph.priority_indices.copy();self.threshold=graph.threshold
        self.ptr=graph.indptr.copy() if method=='O-T' else np.empty(0,np.int64)
        self.indices=graph.indices.copy() if method=='O-T' else np.empty(0,np.int64)
        self.cx=np.asarray(curator,np.float32).copy() if method=='O-G' else np.empty((0,0),np.float32)
        if method=='O-G' and (self.cx.ndim!=2 or len(self.cx)!=len(self.ids)):raise ValueError('O-G requires curator payload')
    def delete(self,ids):
        ids=list(ids)
        if any(not isinstance(i,str) or i not in self.unit_ids for i in ids):raise ValueError('Unknown oracle ID')
        fresh=set(ids)&self.alive
        if len(fresh)>self.horizon:raise ValueError('Cumulative horizon exhausted')
        self.alive.difference_update(fresh);self.horizon-=len(fresh)
        return {'fresh_deletions':len(fresh),'remaining_horizon':self.horizon}
    def selected_indices(self):
        owners=self.ids if self.unit=='record' else self.sources
        alive=np.asarray([u in self.alive for u in owners]);order=[int(i) for i in self.priority if alive[i]]
        if self.method=='O-T':return [i for i in order if not np.any(alive[self.indices[self.ptr[i]:self.ptr[i+1]]])]
        norm={}
        for i in order:
            v=[np.float64(a) for a in self.cx[i]];s=np.float64(0)
            for a in v:s=np.float64(s+np.float64(a*a))
            den=np.sqrt(s);norm[i]=[np.float64(a/den) for a in v]
        selected=[]
        for pos,i in enumerate(order):
            blocked=False
            for j in order[:pos]:
                s=np.float64(0)
                for a,b in zip(norm[i],norm[j]):s=np.float64(s+np.float64(a*b))
                # Do not early stop: count the complete retained pair work.
                if s>self.threshold:blocked=True
            if not blocked:selected.append(i)
        return selected
    def moments(self):
        from ccu.core import ridge_moments
        ix=self.selected_indices();return ridge_moments(self.x[ix],self.y[ix])
    def accounting(self):
        return {'method':self.method,'unit':self.unit,'payload_access':'greater_access_retained_original_arrays_with_live_mask',
                'array_bytes':sum(a.nbytes for a in (self.x,self.y,self.cx,self.priority,self.ptr,self.indices)),
                'remaining_horizon':self.horizon,'forgotten_payload_slots_physically_present':True,
                'maintenance_no_payload_state_claim':False}
    def snapshot(self,path):
        m={'schema':'ccu-oracle-service-1','method':self.method,'unit':self.unit,'ids':self.ids,'sources':self.sources,'alive':sorted(self.alive),'horizon':self.horizon,'threshold':self.threshold}
        if path is None:
            f=io.BytesIO();np.savez(f,metadata=np.asarray(json.dumps(m)),x=self.x,y=self.y,cx=self.cx,priority=self.priority,ptr=self.ptr,indices=self.indices);return f.getvalue()
        with Path(path).open('wb') as f:np.savez(f,metadata=np.asarray(json.dumps(m)),x=self.x,y=self.y,cx=self.cx,priority=self.priority,ptr=self.ptr,indices=self.indices)
        return {'serialized_bytes':Path(path).stat().st_size}
    def snapshot_bytes(self):return self.snapshot(None)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']));obj=cls.__new__(cls)
            for k in ('x','y','cx','priority','ptr','indices'):setattr(obj,k,z[k].copy())
        if m.get('schema')!='ccu-oracle-service-1' or m['method'] not in ('O-G','O-T'):raise ValueError('Invalid oracle schema')
        obj.method=m['method'];obj.unit=m['unit'];obj.ids=tuple(m['ids']);obj.sources=tuple(m['sources']);obj.unit_ids=obj.ids if obj.unit=='record' else tuple(sorted(set(obj.sources)))
        obj.alive=set(m['alive']);obj.horizon=m['horizon'];obj.threshold=m['threshold']
        if not obj.alive<=set(obj.unit_ids) or not 0<=obj.horizon<=len(obj.alive) or len(obj.ids)!=len(obj.x) or len(obj.y)!=len(obj.x):raise ValueError('Invalid oracle state')
        return obj
