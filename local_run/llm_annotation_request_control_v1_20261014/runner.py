"""Capture preparation tests and actual child through exit with bounded logs."""
import contextlib
import importlib.util
import os
from pathlib import Path
import resource
import sys
import time
import unittest
import control as c


def main():
    args = c.parser().parse_args(); prior, h = c.helpers()
    h.verify_package(Path(__file__).resolve().parent, args.control_package_sha256)
    h.verify_package(c.PACKAGE, args.preparation_package_sha256)
    c.require(os.environ.get('SLURM_JOB_ID'), 'Slurm_required')
    root = c.ROOT / 'run'; root.mkdir(mode=0o700, exist_ok=False)
    # The sampler control's helpers verified this earlier frame-control package.
    sys.path.insert(0, str(prior.PRIOR))
    spec = importlib.util.spec_from_file_location('request_bounded_capture', prior.PRIOR / 'runner.py')
    capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)
    os.environ['THIRD_REQUEST_REPO'] = str(c.REPO)
    os.environ['THIRD_REQUEST_PROTOCOL'] = str(c.W / 'tools/model-annotation-v3-20261014')
    os.environ['THIRD_REQUEST_SAMPLER_PACKAGE'] = str(c.W / 'tools/llm-selection-sampler-v1-20261014')
    os.environ['THIRD_PROTOCOL_EVIDENCE'] = str(c.W / 'tools/model-annotation-v3-evidence-v1-20261014')
    sys.path.insert(0, str(c.PACKAGE))
    started = time.perf_counter()
    with (root / 'tests.txt').open('x') as raw:
        log = capture.CappedText(raw, 1024**2)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            suite = unittest.defaultTestLoader.discover(str(c.PACKAGE), pattern='test_*.py')
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    validation = {'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
                  'skipped': len(result.skipped), 'elapsed_seconds': time.perf_counter() - started,
                  'passed': result.wasSuccessful() and result.testsRun == args.expected_tests and not result.skipped}
    h.write_exclusive(root / 'software_validation.json', validation)
    c.require(validation['passed'], 'cluster_tests_failed')
    command = c.child_command(args)
    with (root / 'stdout.txt').open('xb') as out, (root / 'stderr.txt').open('xb') as err:
        started = time.perf_counter()
        rc, size, exceeded = capture.child_capture(command, out, err, 14 * 1024**2)
        elapsed = time.perf_counter() - started
        for stream in (out, err):
            stream.flush(); os.fsync(stream.fileno())
    h.write_exclusive(root / 'process_completion.json', {'command': command, 'returncode': rc,
        'child_wall_seconds_through_exit': elapsed, 'console_bytes': size, 'console_cap_exceeded': exceeded,
        'runner_children_maxrss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'preparation_package_sha256': args.preparation_package_sha256, 'slurm_job_id': os.environ['SLURM_JOB_ID']})
    c.require(not exceeded, 'child_console_cap_exceeded')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
