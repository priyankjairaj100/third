#!/usr/bin/env python3
"""One additive, pinned SciPy install; retain failure evidence and never retry."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request


def require(condition, message):
    if not condition: raise ValueError(message)


def binding(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def normalized(name):
    return re.sub('[-_.]+', '-', name).lower()


def inventory():
    result = {}
    for dist in metadata.distributions():
        name = normalized(dist.metadata['Name'])
        require(name not in result, 'duplicate_installed_distribution')
        record_paths = [Path(dist.locate_file(p)).resolve() for p in (dist.files or [])
                        if str(p).endswith('.dist-info/RECORD')]
        require(len(record_paths) <= 1, 'ambiguous_distribution_RECORD')
        record = record_paths[0] if record_paths else None
        result[name] = {'version': dist.version, 'record_path': str(record) if record else None,
                        'record': binding(record) if record else None}
    return result


def verify_package(folder, sha):
    require(re.fullmatch('[0-9a-f]{64}', sha) is not None, 'invalid_package_SHA')
    require(binding(folder / 'package_manifest.json')['sha256'] == sha, 'repair_package_hash_mismatch')
    for name, expected in json.loads((folder / 'package_manifest.json').read_text())['files'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'unsafe_package_member')
        path = folder / name
        require(path.is_file() and not path.is_symlink() and binding(path) == expected, 'repair_package_member_mismatch')


def verify_source(repo, pins):
    manifest = repo / pins['frozen_code_manifest_relative_path']
    require(binding(manifest)['sha256'] == pins['frozen_code_manifest_sha256'], 'frozen_code_manifest_changed')
    value = json.loads(manifest.read_text())
    for name, expected in value['files'].items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo) and binding(path) == expected, 'frozen_source_changed')
    return binding(manifest)


def run(args):
    require(platform.system() == 'Linux' and os.environ.get('SLURM_JOB_ID'), 'Linux_Slurm_required')
    require(socket.gethostname().split('.')[:2] != ['csis', 'mn1'], 'login_node_prohibited')
    repo, workspace, out = (Path(getattr(args, name)).resolve() for name in ('repo', 'workspace', 'out'))
    require(out.is_relative_to(workspace / 'qualification') and not out.exists(), 'new_private_output_required')
    require(Path(sys.executable).absolute() == workspace / 'envs/encoder-cpu/bin/python', 'exact_encoder_CPU_interpreter_required')
    folder = Path(__file__).resolve().parent
    verify_package(folder, args.package_sha256)
    pins = json.loads((folder / 'expectations.json').read_text())
    require(platform.python_version() == pins['python'] and platform.machine() == pins['machine'], 'pinned_platform_required')
    os.umask(0o077); out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'third-e5-scipy-import-repair-result-v1', 'passed': False, 'stage': 'preflight',
              'package_sha256': args.package_sha256, 'pip_install_attempts': 0,
              'model_forwards': 0, 'frozen_source_modified': False,
              'full_population_cache_acceptance': False, 'confirmatory_study_ready': False}
    start = time.perf_counter()
    before = None
    try:
        report['frozen_source_manifest'] = verify_source(repo, pins)
        before = inventory(); write(out / 'distributions.before.json', before)
        require({k: v['version'] for k, v in before.items()} == pins['preexisting_distribution_versions'],
                'preexisting_distribution_inventory_drift')
        require('scipy' not in before and importlib.util.find_spec('scipy') is None, 'scipy_must_be_absent_no_reinstall')
        sys.path.insert(0, str(repo / 'empirical_execution'))
        from phase4 import embeddings as e
        original_versions = e.installed_versions()
        write(out / 'encoder_versions.before.json', original_versions)
        require(original_versions == pins['frozen_encoder_versions'], 'frozen_encoder_versions_mismatch')
        raw_metadata = folder / 'pypi_scipy_1.17.1.raw.json'
        require(binding(raw_metadata)['sha256'] == pins['official_metadata_sha256'], 'official_metadata_binding_mismatch')
        wheel = pins['wheel']
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
        require(bool(parse_wheel_filename(wheel['filename'])[3] & set(sys_tags())), 'wheel_not_compatible_with_runtime')
        wheels = out / 'wheel'; wheels.mkdir()
        partial, target = wheels / (wheel['filename'] + '.partial'), wheels / wheel['filename']
        report['stage'] = 'single_official_wheel_download'
        # urllib performs one request, without an application retry. Only this
        # immutable published wheel URL is contacted; pip itself stays offline.
        with urllib.request.urlopen(wheel['url'], timeout=60) as response, partial.open('xb') as stream:
            require(response.geturl() == wheel['url'], 'unexpected_wheel_download_redirect')
            copied = 0
            while block := response.read(1024 * 1024):
                copied += len(block); require(copied <= wheel['bytes'], 'wheel_exceeds_declared_size')
                stream.write(block)
            stream.flush(); os.fsync(stream.fileno())
        require(binding(partial) == {k: wheel[k] for k in ('sha256', 'bytes')}, 'published_wheel_hash_or_size_mismatch')
        partial.rename(target)
        write(out / 'downloaded_wheel.json', {'filename': wheel['filename'], 'url': wheel['url'], **binding(target)})
        report['stage'] = 'single_no_dependencies_install'
        command = [sys.executable, '-m', 'pip', '--isolated', 'install', '--only-binary=:all:', '--no-deps',
                   '--no-index', '--no-cache-dir', '--disable-pip-version-check', '--retries', '0',
                   '--require-hashes', '--find-links', str(wheels), '--requirement', str(folder / 'requirements.txt'),
                   '--report', str(out / 'pip-report.json')]
        write(out / 'pip_command.json', {'command': command, 'no_automatic_retry': True})
        report['pip_install_attempts'] = 1
        tick = time.perf_counter()
        with (out / 'pip.stdout.txt').open('x') as stdout, (out / 'pip.stderr.txt').open('x') as stderr:
            process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=False)
            for stream in (stdout, stderr): stream.flush(); os.fsync(stream.fileno())
        write(out / 'pip_process.json', {'returncode': process.returncode, 'seconds_through_exit': time.perf_counter() - tick})
        after = inventory(); write(out / 'distributions.after.json', after)
        require(process.returncode == 0, 'pip_failed_no_retry_or_rollback')
        require(set(after) == set(before) | {'scipy'} and all(after[k] == v for k, v in before.items()),
                'preexisting_distribution_version_or_RECORD_changed')
        require(after['scipy']['version'] == pins['scipy_version'] and after['scipy']['record'] is not None,
                'installed_scipy_identity_missing')
        pip_report = json.loads((out / 'pip-report.json').read_text())
        installs = pip_report.get('install', [])
        require(len(installs) == 1 and normalized(installs[0]['metadata']['name']) == 'scipy'
                and installs[0]['metadata']['version'] == pins['scipy_version']
                and installs[0]['download_info']['archive_info']['hashes']['sha256'] == wheel['sha256'],
                'pip_report_does_not_show_only_pinned_scipy')
        report['stage'] = 'frozen_import_verification'
        importlib.invalidate_caches()
        from phase6 import acceptance
        import scipy
        require(scipy.__version__ == pins['scipy_version'] and Path(scipy.__file__).resolve().is_relative_to(workspace / 'envs/encoder-cpu'),
                'unexpected_scipy_import_origin')
        require(Path(acceptance.__file__).resolve() == repo / 'empirical_execution/phase6/acceptance.py', 'unexpected_frozen_import_origin')
        imported = {}
        for module in tuple(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if path.is_file() and path.is_relative_to(repo):
                    imported[str(path.relative_to(repo))] = binding(path)
        write(out / 'imported_repository_modules.json', imported)
        write(out / 'encoder_versions.after.json', e.installed_versions())
        require(e.installed_versions() == original_versions, 'frozen_encoder_versions_changed')
        verify_source(repo, pins); verify_package(folder, args.package_sha256)
        require(binding(target) == {k: wheel[k] for k in ('sha256', 'bytes')}, 'downloaded_wheel_changed')
        report.update(passed=True, stage='complete', phase6_acceptance_import_passed=True,
                      scipy_version=scipy.__version__, all_preexisting_versions_and_RECORDs_unchanged=True,
                      frozen_encoder_versions_unchanged=True, added_distributions=['scipy'])
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        if before is not None and not (out / 'distributions.after.json').exists():
            try: write(out / 'distributions.after.json', inventory())
            except Exception as evidence_error: report['after_inventory_error'] = str(evidence_error)
    report['elapsed_seconds_before_result_publication'] = time.perf_counter() - start
    report['artifacts'] = {str(path.relative_to(out)): binding(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'workspace', 'out', 'package-sha256'): parser.add_argument('--' + name, required=True)
    try: report = run(parser.parse_args())
    except Exception as error: report = {'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
    print(json.dumps({k: report[k] for k in ('passed', 'stage', 'error_type', 'error') if k in report}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
