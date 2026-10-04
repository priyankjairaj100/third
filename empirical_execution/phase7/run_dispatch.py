"""Local Phase 7 CLI for the unchanged Phase 6 dispatcher.

An explicit policy is validated and forwarded unchanged. Omitting --policy
preserves the Phase 6 default. This wrapper does not grant primary acceptance.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from phase5 import task_program
from phase6 import dispatch

SCHEMA = 'ccu-dispatch-cli-1'
REQUIRED = {'memory_bytes', 'cpu_seconds', 'wall_seconds', 'threads'}
OPTIONAL = {'policy_role', 'convex_certificate_coordinates', 'convex_verifier',
            'state_audit_max_pair_coordinates', 'state_audit_max_coefficient_coordinates',
            'state_audit_absolute_tolerance', 'state_audit_relative_tolerance'}


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Repeated JSON key: ' + key)
        value[key] = item
    return value


def _constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_policy(policy):
    """Validate without default insertion, coercion, or numerical gate weakening."""
    if not isinstance(policy, dict) or not REQUIRED <= set(policy):
        raise ValueError('Policy requires memory_bytes, cpu_seconds, wall_seconds, and threads')
    if set(policy) - REQUIRED - OPTIONAL:
        raise ValueError('Unknown policy fields: ' + ', '.join(sorted(set(policy) - REQUIRED - OPTIONAL)))
    for name, minimum in [('memory_bytes', 64 * 1024**2), ('cpu_seconds', 1), ('threads', 1)]:
        if type(policy[name]) is not int or policy[name] < minimum:
            raise ValueError(name + ' must be an integer at least ' + str(minimum))
    if type(policy['wall_seconds']) not in (int, float) or not math.isfinite(policy['wall_seconds']) or policy['wall_seconds'] <= 0:
        raise ValueError('wall_seconds must be a finite positive number')
    for name in ('convex_certificate_coordinates', 'state_audit_max_pair_coordinates', 'state_audit_max_coefficient_coordinates'):
        if name in policy and (type(policy[name]) is not int or policy[name] < 0):
            raise ValueError(name + ' must be a nonnegative integer')
    for name, maximum in [('state_audit_absolute_tolerance', 1e-11), ('state_audit_relative_tolerance', 1e-10)]:
        if name in policy and (type(policy[name]) not in (int, float) or not math.isfinite(policy[name]) or not 0 <= policy[name] <= maximum):
            raise ValueError(name + ' must be finite, nonnegative, and no weaker than the Phase 6 gate')
    if 'convex_verifier' in policy and policy['convex_verifier'] not in ('fraction_reference', 'exact_dyadic_integer_v1'):
        raise ValueError('Unknown convex verifier')
    if 'policy_role' in policy and (not isinstance(policy['policy_role'], str) or not policy['policy_role'].strip()):
        raise ValueError('policy_role must be a nonempty string')
    return policy


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('registry', 'jobs', 'bundles', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=[dispatch.ENGINEERING_MODE, dispatch.PRIMARY_MODE], default=dispatch.ENGINEERING_MODE)
    p.add_argument('--policy', type=Path, help='Explicit local JSON resource/audit policy; forwarded unchanged. Omission preserves the Phase 6 default.')
    return p


def execute(args):
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError('Preserve existing output: ' + str(args.out))
    # Capture each top-level file exactly once. The receipt hashes the bytes
    # parsed here even if the path is subsequently edited or replaced.
    snapshot = {name: getattr(args, name).read_bytes() for name in ('registry', 'jobs', 'bundles')}
    if args.policy is not None:
        snapshot['policy'] = args.policy.read_bytes()
    policy = None if args.policy is None else validate_policy(loads(snapshot['policy'].decode('utf-8')))
    registry = loads(snapshot['registry'].decode('utf-8'))
    job_bytes = gzip.decompress(snapshot['jobs']) if args.jobs.suffix == '.gz' else snapshot['jobs']
    jobs = [loads(line) for line in job_bytes.decode('utf-8').splitlines() if line.strip()]
    bundles = task_program.load_bound_value(loads(snapshot['bundles'].decode('utf-8')), args.bundles.parent)
    inputs = {name: hashlib.sha256(raw).hexdigest() for name, raw in snapshot.items()}
    receipt = {'schema': SCHEMA, 'input_file_sha256': inputs,
               'wrapper_sha256': file_hash(__file__), 'dispatcher_sha256': file_hash(dispatch.__file__),
               'policy_supplied': args.policy is not None, 'forwarded_policy': policy,
               'omission_uses_unchanged_phase6_default': args.policy is None,
               'top_level_binding_scope': 'Exact captured bytes used for parsing, including compressed job bytes; later edits to those paths do not alter the loaded snapshot.',
               'mode': args.mode, 'planned_jobs': len(jobs),
               'acceptance_scope': 'The unchanged dispatcher retains all primary acceptance gates; this CLI is not an input approval.'}
    result = dispatch.run_dispatch(registry, jobs, bundles, args.out, mode=args.mode,
                                   base_dir=args.bundles.parent, policy=policy)
    # The dispatcher owns directory creation and the complete outcome ledger.
    # This receipt adds the wrapper/file lineage absent from Phase 6 codes().
    with (args.out / 'cli_invocation.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        result = execute(args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        p.exit(2, 'BLOCKED: ' + str(error) + '\n')
    print(json.dumps(result, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
