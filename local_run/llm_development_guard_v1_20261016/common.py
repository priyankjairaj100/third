"""Fixed paths and bounded immutable publication for one development guard."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/nfs_home/users/poonam/ccu-acl2027/repo')
W = REPO.parent / 'local_workspace'
ROOT = W / 'qualification/llm-development-guard-v1-20261016-a01'
BUNDLE = W / 'qualification/llm-development-guard-admission-v1-20261016-a01'
INTAKE = W / 'qualification/llm-development-intake-v1-20261016-a01/run'
CACHE = W / 'qualification/e5-full-sharded-v2-20261013-a01/assembly-run/cache'
FRAME = W / 'qualification/e5-calibration-frame-v1-20261014-a01/run/frame'
SOURCE = W / 'qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu/records.jsonl'
TASK = W / 'qualification/askubuntu-task-calibration-v3-20261016-a01/run'
BENCHMARK = W / 'qualification/llm-guard-benchmark-v1-20261016-a01'
SELECT = W / 'qualification/llm-only-v4-inference-v2-a01'
VALIDATE = W / 'qualification/llm-only-v4-validation-inference-v2-a01'
SELECTION_PIN = 'adcd4704a99bb6d3eabfbd8e5409453a96faa891b00aef93e399e2a23d1034fe'
VALIDATION_PIN = '3b5d8a811909d33cefd9fd6b7c5dbb0e8d17fd1facca447fa0717f1f2ce438c7'
VALIDATION_SAMPLER_PIN = '1abac2e6d14f38c55e78b32031040e7ee387bfe92b8234998187ee1b91e21ad2'
PLAN_PIN = '10ebb3cb8f872c2efaf69306ca5bb23dcec2a015b2747988aafa31485dbbdcae'
REMOTE_FILES = {
    'selection_sampling.json': W/'qualification/llm-only-v4-selection-a01/run/result.json',
    'selection_sample.jsonl': W/'qualification/llm-only-v4-selection-a01/run/selection.private.jsonl',
    'selection_statistics.json': SELECT/'statistics/run/result.json',
    'selection_qwen.json': SELECT/'audit_qwen/run/result.json',
    'selection_mistral.json': SELECT/'audit_mistral/run/result.json',
    'validation_sampling.json': W/'qualification/llm-only-v4-validation-sampling-v2-a01/run/result.json',
    'validation_sample.jsonl': W/'qualification/llm-only-v4-validation-sampling-v2-a01/run/validation.private.jsonl',
    'validation_statistics.json': VALIDATE/'statistics/run/result.json',
    'validation_qwen.json': VALIDATE/'audit_qwen/run/result.json',
    'validation_mistral.json': VALIDATE/'audit_mistral/run/result.json',
}
LOCAL_FILES = ('controller_completion.json', 'selection_review.json', 'validation_review.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def strict(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def seal_check(value):
    require(value['sha256'] == hashlib.sha256(canonical({k: v for k, v in value.items()
                                                      if k != 'sha256'})).hexdigest(), 'canonical_seal')


def byte_binding(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'regular_absolute_file')
    before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    after = path.stat()
    require(all(getattr(before, key) == getattr(after, key) for key in
                ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'hash_input_changed')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def write_bytes(path, raw, *, total_root=None, cap=128 << 20):
    path = Path(path)
    require(len(raw) <= 32 << 20, 'single_file_cap')
    base = total_root or path.parent
    require(sum(p.stat().st_size for p in base.rglob('*') if p.is_file()) + len(raw) <= cap,
            'aggregate_output_cap')
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()


def publish(path, value):
    write_bytes(path, canonical(value) + b'\n', total_root=ROOT if ROOT in path.parents else path.parent)


def verify_package(pin):
    require(binding(HERE/'package_manifest.json')['sha256'] == pin, 'package_pin')
    manifest = strict((HERE/'package_manifest.json').read_bytes())
    require({p.name for p in HERE.iterdir()} == set(manifest['files']) | {'package_manifest.json'},
            'exact_package_inventory')
    for name, wanted in manifest['files'].items():
        require(Path(name).name == name and name != 'package_manifest.json', 'flat_package_member')
        require(binding(HERE/name) == wanted, 'package_member:' + name)
    return manifest
