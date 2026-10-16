"""Separate scan/review processes with bounded logs and per-child resource use."""
import contextlib
import os
import signal
import selectors
import subprocess
import sys
import time
import unittest
import common as c


def child(label, command, pin):
    started = time.perf_counter(); size = 0; exceeded = False
    with (c.ROOT/(label+'.stdout.txt')).open('xb') as out, (c.ROOT/(label+'.stderr.txt')).open('xb') as err:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, out); selector.register(process.stderr, selectors.EVENT_READ, err)
        try:
            while selector.get_map():
                for key, _ in selector.select(timeout=1):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj); key.fileobj.close(); continue
                    if size+len(data) > 4 << 20:
                        try:
                            os.kill(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        exceeded = True; data = data[:max(0, (4 << 20)-size)]
                    key.data.write(data); size += len(data)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        finally:
            selector.close()
            if process.returncode is None:
                process.kill(); process.wait()
        for stream in (out, err):
            stream.flush(); os.fsync(stream.fileno())
    receipt = {'job_id': os.environ['SLURM_JOB_ID'], 'command': command, 'returncode': process.returncode,
               'console_bytes': size, 'console_cap_exceeded': exceeded,
               'wall_seconds_through_child_exit': time.perf_counter()-started,
               'child_peak_rss_kib_linux': usage.ru_maxrss,
               'package_sha256': pin, 'resource_accounting': 'os.wait4 for this exact child; not cumulative sibling maximum'}
    c.publish(c.ROOT/(label+'_process.json'), receipt)
    c.require(not exceeded and process.returncode == 0, 'child_failed:'+label)


def main():
    pin, admitted = sys.argv[1:]; c.verify_package(pin)
    c.require(c.ROOT.is_dir() and os.environ.get('SLURM_JOB_ID'), 'launched_root')
    os.environ['THIRD_LOCAL_REPO'] = str(c.REPO)
    with (c.ROOT/'software_tests.txt').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.discover(str(c.HERE), pattern='test_*.py'))
    expected = c.strict((c.HERE/'SOFTWARE_VALIDATION.json').read_bytes())['tests']
    c.publish(c.ROOT/'software_validation.json', {'passed': result.wasSuccessful(), 'tests': result.testsRun,
              'errors': len(result.errors), 'failures': len(result.failures), 'skipped': len(result.skipped)})
    c.require(result.wasSuccessful() and result.testsRun == expected and not result.skipped, 'software_preflight')
    base = ['--package-sha256', pin, '--admission-sha256', admitted]
    for mode in ('primary', 'reference'):
        child(mode, [sys.executable, '-B', str(c.HERE/'worker.py'), '--mode', mode, *base], pin)
    child('review', [sys.executable, '-B', str(c.HERE/'review.py'), *base], pin)
    c.publish(c.ROOT/'pipeline_completion.json', {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'],
              'package_sha256': pin, 'admission_sha256': admitted, 'result': c.binding(c.ROOT/'result.json'),
              'external_terminal_accounting_required': True, 'effect_experiments_started': False})


if __name__ == '__main__':
    main()
