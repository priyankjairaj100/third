"""Defined logical measurements and simultaneous process-tree RSS samples.

No counter here measures physical memory traffic or exact allocator attribution.
"""
import hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
import numpy as np

def state_inventory(state):
    """Exact numeric-buffer extents and canonical metadata representation size.

    The metadata representation is a defined JSON wire form. It is not Python
    allocator memory or a claim that the algorithm read every metadata byte.
    """
    seen={};buffers={};arrays=[]
    def walk(value,path):
        if isinstance(value,np.ndarray):
            root=value
            while isinstance(root.base,np.ndarray):root=root.base
            key=id(root);buffers[key]=root.nbytes
            arrays.append({'path':path,'dtype':str(value.dtype),'shape':list(value.shape),'logical_bytes':value.nbytes})
            return {'array':len(arrays)-1}
        if value is None or isinstance(value,(str,int,float,bool)):return value
        if isinstance(value,np.generic):return value.item()
        ident=id(value)
        if ident in seen:return {'reference':seen[ident]}
        seen[ident]=path
        if isinstance(value,dict):return {'mapping':[[walk(k,path+'.key'),walk(v,path+'.'+str(k))] for k,v in sorted(value.items(),key=lambda kv:repr(kv[0]))]}
        if isinstance(value,(tuple,list,set,frozenset)):
            seq=sorted(value,key=repr) if isinstance(value,(set,frozenset)) else value
            return {'kind':type(value).__name__,'values':[walk(v,path+'.'+str(i)) for i,v in enumerate(seq)]}
        if hasattr(value,'__dict__'):return {'class':type(value).__module__+'.'+type(value).__name__,'fields':walk(value.__dict__,path)}
        if hasattr(value,'__slots__'):return {'class':type(value).__name__,'fields':{s:walk(getattr(value,s),path+'.'+s) for s in value.__slots__ if hasattr(value,s)}}
        raise TypeError('Unaccounted state object: '+type(value).__name__)
    metadata=walk(state,'state');raw=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {'schema':'ccu-logical-state-inventory-1','unique_ndarray_root_buffer_bytes':sum(buffers.values()),
        'ndarray_views_logical_bytes':sum(a['logical_bytes'] for a in arrays),'arrays':arrays,
        'canonical_metadata_utf8_bytes':len(raw),'canonical_metadata_sha256':hashlib.sha256(raw).hexdigest(),
        'python_allocator_bytes':None,'hardware_memory_traffic_bytes':None,
        'metadata_scope':'defined_JSON_wire_representation_including_type_array_and_alias_descriptors_not_heap_size_or_access_traffic'}

def serialized_inventory(parts):
    """Actual uncompressed NPZ members, headers, and sidecar file sizes."""
    out=[]
    for name,raw in parts.items():
        row={'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        if name.endswith('.npz'):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                row['members']=[{'name':i.filename,'uncompressed_bytes':i.file_size,'compressed_bytes':i.compress_size} for i in z.infolist()]
                row['container_overhead_bytes']=len(raw)-sum(i.compress_size for i in z.infolist())
        out.append(row)
    return {'files':out,'simultaneous_full_output_buffer_bytes':sum(len(raw) for raw in parts.values()),
            'buffer_size_is_not_total_serialization_peak':True}

def logical_update_meter(state,op):
    """Count stated materializations; do not infer uninstrumented operations."""
    ops=op.get('operations',{});rows=ops.get('payload_rows_read');out={
        'scope':'explicit_signed_BLAS_payload_gathers_and_known_compaction_copies_only',
        'all_algorithm_memory_accesses_counted':False,'metadata_read_traffic_bytes':None,
        'hardware_memory_traffic_bytes':None,'compaction_copy_output_bytes':None,
        'payload_input_value_bytes':None,'gather_and_precision_copy_output_bytes':None}
    if rows is not None:
        d,c=int(state.dimension),int(state.outputs);fp32=state.gram.dtype==np.float32
        out['payload_input_value_bytes']=int(rows)*(4*d+8*c)
        out['gather_and_precision_copy_output_bytes']=int(rows)*((4*d+8*c+4*c) if fp32 else (4*d+8*c+8*d))
        # Historical compaction does advanced indexing then .copy for each field.
        out['compaction_copy_output_bytes']=2*sum(getattr(state,k).nbytes for k in ('x','y','owner','counts','selected')) if op.get('compacted') else 0
        out['metadata_new_array_writes_excluded']=True
    elif getattr(state,'method',None)=='B-F':
        out['scope']='B-F_survivor_payload_advanced_index_materialization'
        out['compaction_copy_output_bytes']=op['payload_copy_output_bytes']
    return out

def process_tree_sample(root_pid):
    """One sweep: sum RSS at observed times for root and descendants.

    Samples are not atomic. Shared mappings are counted once per process.
    Short-lived processes and peaks between samples can be missed.
    """
    records={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:
            raw=(entry/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
            records[int(entry.name)]=(int(tail[1]),int(tail[21]))
        except (OSError,ValueError,IndexError):continue
    host_self=int(Path('/proc/self/stat').read_text().split(' ',1)[0]);host_root=None
    # Managed runtimes can mount /proc from an ancestor PID namespace.
    # Resolve only our direct child, then use the observed parent graph.
    for pid,(parent,_) in records.items():
        if parent!=host_self:continue
        try:
            status=Path(f'/proc/{pid}/status').read_text()
            ns=[int(v) for line in status.splitlines() if line.startswith('NSpid:') for v in line.split()[1:]]
            if pid==root_pid or root_pid in ns:host_root=pid;break
        except (OSError,ValueError):continue
    selected={host_root} if host_root is not None else set();changed=True
    while changed:
        changed=False
        for pid,(parent,_) in records.items():
            if parent in selected and pid not in selected:selected.add(pid);changed=True
    selected&=records.keys();page=os.sysconf('SC_PAGE_SIZE')
    return {'rss_sum_bytes':sum(records[p][1]*page for p in selected),'observed_pids':sorted(selected),'namespace_root_pid':root_pid,'proc_root_pid':host_root}

def imported_dependency_manifest(file_hash):
    """Hash imported Python files and state the finite observed closure."""
    began=time.perf_counter();paths=set()
    for module in tuple(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if isinstance(p,str):
            path=Path(p)
            if path.is_file():paths.add(str(path.resolve()))
    files=[{'path':p,'sha256':file_hash(p),'bytes':Path(p).stat().st_size} for p in sorted(paths)]
    return {'schema':'ccu-imported-module-files-1','files':files,'seconds':time.perf_counter()-began,
        'scope':'actually_imported_module_files_at_measurement_boundary_plus_separate_mapped_ELF_manifest',
        'full_transitive_distribution_or_build_dependency_closure':False,
        'dynamic_libraries_loaded_after_boundary_may_be_absent':True,
        'new_file_opens_during_repair_are_kernel_denied':True}

class SnapshotReadMeter:
    """Measure returned bytes from Python binary reads of own NPZ snapshots.

    Categories use disjoint byte ranges in stored ZIP members. Repeated reads
    count repeatedly. This excludes kernel read-ahead and unobserved C reads.
    """
    def __init__(self,paths):
        import struct
        self.ranges={};self.counts={'payload_array_bytes':0,'statistic_array_bytes':0,
            'metadata_array_bytes':0,'serialization_format_bytes':0};self.calls=0
        self.indexed_file_bytes=0
        for source in paths:
            path=Path(source).resolve()
            if path.suffix!='.npz':continue
            spans=[];self.indexed_file_bytes+=path.stat().st_size
            with path.open('rb') as raw,zipfile.ZipFile(raw) as archive:
                for member in archive.infolist():
                    if member.compress_type!=zipfile.ZIP_STORED:raise ValueError('Read categories require uncompressed snapshot members')
                    raw.seek(member.header_offset);header=raw.read(30);name_len,extra_len=struct.unpack_from('<HH',header,26)
                    offset=member.header_offset+30+name_len+extra_len
                    with archive.open(member) as item:
                        version=np.lib.format.read_magic(item)
                        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(item)
                        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(item)
                        else:raise ValueError('Unsupported NPY member header')
                        start=offset+item.tell()
                    name=member.filename.removesuffix('.npy')
                    if name in ('x','y','features','targets','curator','cx'):category='payload_array_bytes'
                    elif dtype.kind=='f':category='statistic_array_bytes'
                    else:category='metadata_array_bytes'
                    spans.append((start,offset+member.file_size,category))
            self.ranges[str(path)]=spans
    def __enter__(self):
        import builtins
        self.original_builtin=builtins.open;self.original_io=io.open;meter=self
        class Stream:
            def __init__(self,raw,path):self.raw=raw;self.path=path
            def _record(self,start,size):
                if not size:return
                meter.calls+=1;covered=0
                for lo,hi,category in meter.ranges[self.path]:
                    overlap=max(0,min(start+size,hi)-max(start,lo));meter.counts[category]+=overlap;covered+=overlap
                meter.counts['serialization_format_bytes']+=size-covered
            def read(self,*args):
                start=self.raw.tell();value=self.raw.read(*args);self._record(start,len(value));return value
            def readinto(self,buffer):
                start=self.raw.tell();size=self.raw.readinto(buffer);self._record(start,size or 0);return size
            def readline(self,*args):
                start=self.raw.tell();value=self.raw.readline(*args);self._record(start,len(value));return value
            def __getattr__(self,name):return getattr(self.raw,name)
            def __enter__(self):return self
            def __exit__(self,*args):return self.raw.__exit__(*args)
        def wrap(original):
            def opener(file,mode='r',*args,**kwargs):
                stream=original(file,mode,*args,**kwargs)
                if isinstance(file,(str,bytes,os.PathLike)) and 'b' in mode and 'r' in mode:
                    path=str(Path(file).resolve())
                    if path in meter.ranges:return Stream(stream,path)
                return stream
            return opener
        builtins.open=wrap(self.original_builtin);io.open=wrap(self.original_io);return self
    def __exit__(self,*args):
        import builtins
        builtins.open=self.original_builtin;io.open=self.original_io
    def report(self):return {'schema':'ccu-own-snapshot-python-read-ranges-1','returned_binary_read_bytes_by_category':self.counts,
        'returned_binary_read_bytes_total':sum(self.counts.values()),'read_calls':self.calls,'indexed_snapshot_logical_bytes':self.indexed_file_bytes,
        'scope':'Python_binary_read_return_values_for_own_uncompressed_NPZ_load_only',
        'hashing_reads_excluded':True,'text_sidecar_reads_excluded':True,'C_level_bypasses_not_observed':True,
        'kernel_storage_or_memory_traffic_claim':False}
