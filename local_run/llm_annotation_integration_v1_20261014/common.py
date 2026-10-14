"""Small strict byte/package helpers. No entropy, network, model loading or jobs."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
def require(ok,why):
    if not ok: raise ValueError(why)
def canonical(x,ascii=False):
    return json.dumps(x,ensure_ascii=ascii,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate_json_key');out[k]=v
        return out
    def bad(x):raise ValueError('nonfinite_json')
    def floating(x):
        v=float(x);require(math.isfinite(v),'nonfinite_json');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=floating)
def read(path):return strict(Path(path).read_bytes())
def binding(path):
    h=hashlib.sha256();n=0
    with Path(path).open('rb') as f:
        for raw in iter(lambda:f.read(1<<20),b''):n+=len(raw);h.update(raw)
    return {'bytes':n,'sha256':h.hexdigest()}
def valid_binding(value):
    require(type(value) is dict and set(value)=={'bytes','sha256'},'binding_fields')
    require(type(value['bytes']) is int and value['bytes']>=0,'binding_bytes')
    h=value['sha256'];require(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'binding_sha')
    return value
def seal(obj):
    require(type(obj) is dict and obj.get('sha256')==sha(canonical({k:v for k,v in obj.items() if k!='sha256'})),'internal_seal')
def sealed(obj):return dict(obj,sha256=sha(canonical(obj)))
def pins():return read(HERE/'expectations.json')
class Watch:
    def __init__(self):self.files={};self.dirs={}
    def file(self,path,want=None,cap=64*1024**2):
        p=Path(path).absolute();require(p.is_file() and not p.is_symlink(),'regular_file_required:'+str(p))
        require(p.stat().st_size<=cap,'file_size_cap')
        b=binding(p);require(want is None or b==valid_binding(want),'file_hash:'+str(p));self.files[str(p)]=b;return b
    def inventory(self,path,names):
        p=Path(path);require(p.is_dir() and not p.is_symlink(),'regular_directory')
        require(all(not x.is_symlink() for x in p.rglob('*')),'directory_symlink')
        actual={str(x.relative_to(p)) for x in p.rglob('*') if x.is_file()}
        require(actual==set(names),'directory_inventory:'+str(p));self.dirs[str(p)]=sorted(names)
    def recheck(self):
        for p,b in list(self.files.items()):self.file(p,b,max(64*1024**2,b['bytes']))
        for p,names in self.dirs.items():self.inventory(p,names)
def package(path,want,watch=None):
    w=watch or Watch();p=Path(path);w.file(p/'package_manifest.json',want,1<<20)
    m=read(p/'package_manifest.json');require(type(m.get('files')) is dict,'package_files')
    for name,b in m['files'].items():
        require(type(name) is str and Path(name).name==name and name not in ('.','..','package_manifest.json'),'package_member')
        w.file(p/name,b)
    w.inventory(p,set(m['files'])|{'package_manifest.json'});return p
def load(name,path):
    p=Path(path);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def jsonl(path):
    out=[]
    with Path(path).open('rb') as f:
        for raw in f:
            require(raw.endswith(b'\n') and len(raw)<=16*1024**2 and raw.strip(),'physical_jsonl')
            out.append(strict(raw))
    return out
def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def new_json(path,obj):
    p=Path(path);raw=canonical(obj)+b'\n';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fsync_dir(p.parent);return binding(p)
