"""Bounded maintenance of old, unshared pip HTTP download-cache pairs only."""
import datetime, hashlib, json, os, re, socket, stat, sys, time
from pathlib import Path

CACHE = Path('/nfs_home/users/poonam/.cache/pip/http-v2')
ROOT = Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace/maintenance/pip-cache-cleanup-20261015-a01')
AGE = 7 * 86400
TARGET = 2 * 1024**3

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def record(st):
    return dict(dev=st.st_dev, ino=st.st_ino, mode=st.st_mode, uid=st.st_uid,
                nlink=st.st_nlink, size=st.st_size, blocks=st.st_blocks,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)

def valid_name(name):
    parts = Path(name).parts
    return (len(parts) == 6 and all(re.fullmatch('[0-9a-f]', x) for x in parts[:5])
            and re.fullmatch('[0-9a-f]{56}(?:\\.body)?', parts[5]) is not None
            and ''.join(parts[:5]) == parts[5][:5])

def eligible(st, cutoff, dev):
    return (stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_uid == os.getuid()
            and st.st_dev == dev and st.st_mtime < cutoff and st.st_ctime < cutoff)

def save(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

def parent_fd(name):
    assert valid_name(name)
    fd = os.open(str(CACHE), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in Path(name).parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd); raise

def main():
    assert os.environ.get('SLURM_JOB_ID') and socket.gethostname() in ('csis.cn1', 'cn1')
    assert CACHE.resolve() == CACHE and ROOT.resolve() == ROOT
    assert not CACHE.is_symlink() and CACHE.is_dir()
    dev = CACHE.stat().st_dev
    assert dev == ROOT.stat().st_dev
    cutoff = time.time() - AGE
    mode = sys.argv[1]
    if mode == 'inspect':
        start = time.monotonic(); stack = [CACHE]; candidates = []; seen = 0; allocated = 0
        while stack:
            assert time.monotonic() - start < 240, 'bounded scan deadline'
            folder = stack.pop()
            with os.scandir(folder) as entries:
                for e in entries:
                    seen += 1; assert seen <= 100000, 'bounded scan entry limit'
                    st = e.stat(follow_symlinks=False)
                    if stat.S_ISDIR(st.st_mode) and st.st_dev == dev:
                        stack.append(Path(e.path)); continue
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    allocated += st.st_blocks * 512
                    rel = str(Path(e.path).relative_to(CACHE))
                    if not rel.endswith('.body') or not valid_name(rel) or st.st_size < 1024**2:
                        continue
                    meta = Path(e.path[:-5])
                    if not meta.exists() or meta.is_symlink():
                        continue
                    mst = meta.lstat()
                    if eligible(st, cutoff, dev) and eligible(mst, cutoff, dev):
                        candidates.append({'files': [{'name': rel, 'stat': record(st)},
                                                    {'name': str(meta.relative_to(CACHE)), 'stat': record(mst)}],
                                           'allocated_bytes': (st.st_blocks + mst.st_blocks) * 512})
        candidates.sort(key=lambda x: (-x['allocated_bytes'], x['files'][0]['name']))
        selected = []; total = 0
        for c in candidates:
            selected.append(c); total += c['allocated_bytes']
            if total >= TARGET:
                break
        plan = dict(schema='third-pip-cache-cleanup-v1', mode=mode, created_utc=now(),
                    job_id=os.environ['SLURM_JOB_ID'], cache=str(CACHE), min_age_seconds=AGE,
                    target_bytes=TARGET, inspected_entries=seen, cache_allocated_bytes=allocated,
                    eligible_pairs=len(candidates), eligible_bytes=sum(x['allocated_bytes'] for x in candidates),
                    selected_pairs=selected, selected_allocated_bytes=total,
                    scientific_artifacts_touched=False, installed_environments_touched=False)
        save('plan.json', plan)
        print(json.dumps({k:v for k,v in plan.items() if k != 'selected_pairs'}, sort_keys=True))
    elif mode == 'apply':
        raw = (ROOT / 'plan.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sys.argv[2], 'exact reviewed plan'
        plan = json.loads(raw)
        assert plan['cache'] == str(CACHE) and plan['min_age_seconds'] == AGE and plan['selected_pairs']
        assert not (ROOT / 'apply_intent.json').exists(), 'no automatic cleanup retry'
        # Validate the entire exact plan before changing anything.
        for pair in plan['selected_pairs']:
            for f in pair['files']:
                fd = parent_fd(f['name'])
                try:
                    st = os.stat(Path(f['name']).name, dir_fd=fd, follow_symlinks=False)
                    assert record(st) == f['stat'] and eligible(st, cutoff, dev)
                finally:
                    os.close(fd)
        save('apply_intent.json', dict(created_utc=now(), plan_sha256=sys.argv[2], job_id=os.environ['SLURM_JOB_ID']))
        deleted = 0; total = 0
        with (ROOT / 'deleted.jsonl').open('x') as journal:
            for pair in plan['selected_pairs']:
                for f in pair['files']:
                    fd = parent_fd(f['name'])
                    try:
                        st = os.stat(Path(f['name']).name, dir_fd=fd, follow_symlinks=False)
                        assert record(st) == f['stat'] and eligible(st, cutoff, dev), 'cache entry changed'
                        os.unlink(Path(f['name']).name, dir_fd=fd)
                        deleted += 1; total += st.st_blocks * 512
                        journal.write(json.dumps(dict(name=f['name'], removed_allocated_bytes=st.st_blocks*512))+'\n')
                        journal.flush(); os.fsync(journal.fileno())
                    finally:
                        os.close(fd)
        result = dict(schema='third-pip-cache-cleanup-result-v1', completed_utc=now(),
                      job_id=os.environ['SLURM_JOB_ID'], plan_sha256=sys.argv[2], deleted_files=deleted,
                      removed_allocated_bytes=total, actual_quota_reclaim_requires_fresh_login_check=True,
                      scientific_artifacts_touched=False, installed_environments_touched=False)
        save('result.json', result); print(json.dumps(result, sort_keys=True))
    else:
        raise ValueError('unknown mode')

if __name__ == '__main__':
    main()
