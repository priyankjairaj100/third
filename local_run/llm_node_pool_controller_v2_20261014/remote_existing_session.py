"""Use this turn's explicitly selected, authenticated master without retry."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--timeout', type=int, default=60)
args = parser.parse_args()
assert 1 <= args.timeout <= 240
control = Path(__file__).with_name('agent_control_path.txt').read_text().strip()
assert control.startswith('/private/tmp/third-csis-20261014-') and control.endswith('/master')
script = sys.stdin.read()
check = subprocess.run(['ssh', '-S', control, '-o', 'BatchMode=yes', '-O', 'check', 'csis'], capture_output=True, text=True, timeout=5)
if check.returncode:
    print('Explicit authenticated SSH master unavailable; no remote command attempted.', file=sys.stderr)
    print(check.stderr, end='', file=sys.stderr)
    raise SystemExit(75)
try:
    result = subprocess.run(['ssh', '-S', control, '-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
        '-o', 'ProxyCommand=false', '-o', 'ConnectTimeout=10', 'csis', 'python3 -'],
        input=script, capture_output=True, text=True, timeout=args.timeout)
except subprocess.TimeoutExpired:
    print('SSH operation timed out. Stop; reconcile remote receipts before any further mutation.', file=sys.stderr)
    raise SystemExit(124)
print(result.stdout, end='')
print(result.stderr, end='', file=sys.stderr)
raise SystemExit(result.returncode)
