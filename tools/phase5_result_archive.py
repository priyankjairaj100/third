#!/usr/bin/env python3
"""Lossless archive/restore of verbose Phase5 per-process result artifacts.

The archive contains only regular files under the designated jobs directory.
Hashes are checked before extraction and existing differing files are refused.
"""
from pathlib import Path,PurePosixPath
import argparse,gzip,hashlib,io,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'empirical_execution/phase5/results/isolated_natural_engineering_final'
ARCHIVE=RESULT/'jobs.tar.gz';MANIFEST=RESULT/'jobs_archive.json'

def sha(raw):return hashlib.sha256(raw).hexdigest()

def pack():
    if ARCHIVE.exists() or MANIFEST.exists():raise FileExistsError('Archive exists; preserve it')
    paths=sorted((RESULT/'jobs').rglob('*'));files=[]
    for p in paths:
        if p.is_symlink():raise ValueError('Symlinks are not archived')
        if p.is_file():files.append(p)
    if not files:raise ValueError('No results found')
    entries=[]
    with ARCHIVE.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for p in files:
                    content=p.read_bytes();name=p.relative_to(RESULT).as_posix()
                    info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o644;info.mtime=0
                    tar.addfile(info,io.BytesIO(content))
                    entries.append({'path':name,'bytes':len(content),'sha256':sha(content)})
    result={'schema':'ccu-lossless-result-archive-1','archive':'jobs.tar.gz',
        'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha(ARCHIVE.read_bytes()),
        'file_count':len(entries),'uncompressed_file_bytes':sum(e['bytes'] for e in entries),
        'files':entries,'omitted_results':False,'source_hash':sha(Path(__file__).read_bytes())}
    MANIFEST.write_text(json.dumps(result,indent=2)+'\n')
    return {'file_count':len(entries),'archive_bytes':result['archive_bytes']}

def verify_restore(restore=False):
    m=json.loads(MANIFEST.read_text());raw=ARCHIVE.read_bytes()
    if sha(raw)!=m['archive_sha256'] or len(raw)!=m['archive_bytes']:raise ValueError('Archive checksum mismatch')
    expected={e['path']:e for e in m['files']}
    if len(expected)!=m['file_count']:raise ValueError('Manifest duplicate/file-count mismatch')
    seen=set();contents=[]
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
        for member in tar:
            name=member.name;p=PurePosixPath(name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='jobs' or name in seen or name not in expected:
                raise ValueError('Invalid archive member')
            data=tar.extractfile(member).read();entry=expected[name]
            if len(data)!=entry['bytes'] or sha(data)!=entry['sha256']:raise ValueError('Member checksum mismatch')
            target=RESULT/name
            if target.exists() and (target.is_symlink() or not target.is_file() or sha(target.read_bytes())!=entry['sha256']):
                raise ValueError('Existing result differs; never overwrite')
            # No extraction through a symlinked directory.
            if any(parent.is_symlink() for parent in target.parents if parent!=ROOT.parent):raise ValueError('Symlinked destination')
            seen.add(name);contents.append((target,data))
    if seen!=set(expected):raise ValueError('Missing archive members')
    if restore:
        for target,data in contents:
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return {'verified_files':len(seen),'restored':restore,'all_bytes_verified':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','verify','restore']);a=p.parse_args()
    print(json.dumps(pack() if a.action=='pack' else verify_restore(a.action=='restore'),indent=2))
