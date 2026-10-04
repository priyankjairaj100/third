"""Read-only arithmetic preflight of the frozen native audit and verifier caps.

No arrays, corpus records, graph, worker, optimizer, or verifier are executed.
Only the optional new report is written. Count feasibility is not feasibility
under a time or memory limit, and this command does not issue a development lock.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'ccu-native-resource-preflight-1'
PAIR_CAP = 'state_audit_max_pair_coordinates'
COEFFICIENT_CAP = 'state_audit_max_coefficient_coordinates'
CONVEX_CAP = 'convex_certificate_coordinates'
CAPS = (PAIR_CAP, COEFFICIENT_CAP, CONVEX_CAP)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Repeated JSON key: ' + key)
        value[key] = item
    return value


def _constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def loads(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer at least ' + str(minimum))
    return value


def validate_policy(policy):
    """Accept the explicit dispatcher policy shape, without mutating its value."""
    required = {'memory_bytes', 'cpu_seconds', 'wall_seconds', 'threads'}
    optional = set(CAPS) | {'policy_role', 'convex_verifier',
        'state_audit_absolute_tolerance', 'state_audit_relative_tolerance'}
    if not isinstance(policy, dict) or not required <= set(policy):
        raise ValueError('Policy requires memory_bytes, cpu_seconds, wall_seconds, and threads')
    if set(policy) - required - optional:
        raise ValueError('Unknown policy fields: ' + ', '.join(sorted(set(policy) - required - optional)))
    for name, minimum in [('memory_bytes', 64 * 1024**2), ('cpu_seconds', 1), ('threads', 1)]:
        integer(policy[name], name, minimum)
    wall = policy['wall_seconds']
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError('wall_seconds must be a finite positive number')
    for name in CAPS:
        if name in policy:
            integer(policy[name], name)
    for name, maximum in [('state_audit_absolute_tolerance', 1e-11), ('state_audit_relative_tolerance', 1e-10)]:
        if name in policy:
            value = policy[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(name + ' must not weaken the frozen numerical gate')
    if 'convex_verifier' in policy and policy['convex_verifier'] not in ('fraction_reference', 'exact_dyadic_integer_v1'):
        raise ValueError('Unknown convex verifier')
    if 'policy_role' in policy and (not isinstance(policy['policy_role'], str) or not policy['policy_role'].strip()):
        raise ValueError('policy_role must be a nonempty string')
    return policy


def _function(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError('Frozen source contract unavailable: function ' + name)
    return matches[0]


def _same(a, expression):
    return ast.dump(a, include_attributes=False) == ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False)


def source_contract(source_bytes=None):
    """Capture source once; refuse unsupported formulas instead of guessing.

    AST matching authenticates the particular cap formulas, not the complete
    program's semantics. Whole-file hashes bind the inspected source bytes.
    No source code is imported or executed.
    """
    names = ('phase6/state_audit.py', 'phase4/convex.py', 'phase5/convex_multioutput.py',
             'phase6/dyadic_convex.py', 'phase6/dispatch.py', 'phase6/recipes.py',
             'phase7/run_dispatch.py')
    raw = {name: (ROOT / name).read_bytes() for name in names} if source_bytes is None else dict(source_bytes)
    if set(raw) != set(names):
        raise ValueError('Complete frozen source snapshot required')
    trees = {name: ast.parse(value.decode('utf-8')) for name, value in raw.items()}
    evidence = []

    def require(name, function, expression, kind='guard'):
        node = _function(trees[name], function)
        candidates = [n.test for n in ast.walk(node) if isinstance(n, ast.If)] if kind == 'guard' else [
            n.value for n in ast.walk(node) if isinstance(n, ast.Assign)]
        matches = [n for n in candidates if _same(n, expression)]
        if len(matches) != 1:
            raise ValueError('Unsupported frozen ' + kind + ': ' + name + ':' + function)
        evidence.append({'path': name, 'function': function, 'line': matches[0].lineno,
                         'kind': kind, 'expression': ast.unparse(matches[0])})

    require('phase6/state_audit.py', 'direct_graph', 'n * (n - 1) // 2 * x.shape[1] > maximum_pair_coordinates')
    require('phase6/state_audit.py', 'independent_coefficients', 'd * (d + 1) // 2 + d * c + 1', 'assignment')
    require('phase6/state_audit.py', 'independent_coefficients', 'len(symbols) * packed > maximum_coordinates')
    require('phase6/state_audit.py', 'audit_service_state', "method in ('P-I', 'P-S', 'P-R') and full_state")
    for name in ('phase4/convex.py', 'phase6/dyadic_convex.py'):
        require(name, 'certify_logistic', 'n * d > cap')
    for name in ('phase5/convex_multioutput.py', 'phase6/dyadic_convex.py'):
        require(name, 'certify_multioutput', 'x.size * y.shape[1] > cap')

    def policy_default(name, function, key):
        matches = [node for node in ast.walk(_function(trees[name], function))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'policy'
            and node.func.attr == 'get' and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == key]
        if len(matches) != 1:
            raise ValueError('Frozen policy default unavailable: ' + key)
        value = integer(ast.literal_eval(matches[0].args[1]), key)
        evidence.append({'path': name, 'function': function, 'line': matches[0].lineno,
                         'kind': 'policy_default', 'key': key, 'value': value})
        return value

    defaults = {
        PAIR_CAP: policy_default('phase6/state_audit.py', 'audit_service_state', PAIR_CAP),
        COEFFICIENT_CAP: policy_default('phase6/state_audit.py', 'audit_service_state', COEFFICIENT_CAP),
        CONVEX_CAP: policy_default('phase6/dispatch.py', 'convex_job', CONVEX_CAP)}
    for name, function in [('phase4/convex.py', 'certify_logistic'),
            ('phase5/convex_multioutput.py', 'certify_multioutput'),
            ('phase6/dyadic_convex.py', 'certify_logistic'),
            ('phase6/dyadic_convex.py', 'certify_multioutput')]:
        args = _function(trees[name], function).args
        values = {arg.arg: ast.literal_eval(value) for arg, value in zip(args.kwonlyargs, args.kw_defaults)
                  if arg.arg == 'max_coordinates'}
        if values.get('max_coordinates') != defaults[CONVEX_CAP]:
            raise ValueError('Verifier and dispatcher defaults disagree')
    return {'source_sha256': {name: sha256(value) for name, value in raw.items()},
            'default_caps': defaults, 'source_expressions': evidence,
            'scope': 'Inspected cap expressions and defaults; complete source hashes; no source execution.'}


def maximum_pair_rows(cap, dimension):
    """Largest N with C(N,2)*dimension <= cap, including zero-cap N=1."""
    return (1 + math.isqrt(1 + 8 * (cap // dimension))) // 2


def compare_count(demand, lower, upper, cap, label):
    if demand is not None:
        status = 'passes_count_cap' if demand <= cap else 'refused_by_count_cap'
    elif lower > cap:
        status = 'refused_by_count_cap'
    elif upper is not None and upper <= cap:
        status = 'passes_count_cap_for_all_counts_in_bound'
    else:
        status = 'unknown_count_required'
    return {'status': status, 'exact_demand': demand, 'lower_bound': lower,
            'upper_bound': upper, 'cap': cap, 'unit': label,
            'exact_cap_equality_passes': True, 'runtime_or_memory_qualification': False}


def preflight(*, records, dimension, outputs, curator_dimension=None, retained_records=None,
              selected_records=None, symbolic_keys=None, audit_scope='full', policy=None,
              contract=None):
    """Inspect one planned panel and one potential retained checkpoint shape.

    symbolic_keys means len(fresh_symbols), not horizon, stored basis size,
    numeric nonzero support, or the actual/fresh union used during comparison.
    """
    n = integer(records, 'records')
    d = integer(dimension, 'dimension', 1)
    c = d if curator_dimension is None else integer(curator_dimension, 'curator_dimension', 1)
    r = n if retained_records is None else integer(retained_records, 'retained_records')
    if r > n:
        raise ValueError('retained_records cannot exceed original records')
    if not isinstance(outputs, (list, tuple)) or not outputs or len(set(outputs)) != len(outputs):
        raise ValueError('Distinct positive output counts required')
    for q in outputs:
        integer(q, 'outputs', 1)
    if selected_records is not None and integer(selected_records, 'selected_records') > r:
        raise ValueError('selected_records cannot exceed retained_records')
    if symbolic_keys is not None and integer(symbolic_keys, 'symbolic_keys') > 2 * r:
        raise ValueError('symbolic_keys exceeds two monomials per retained record')
    if audit_scope not in ('full', 'light'):
        raise ValueError('audit_scope must be full or light')
    if policy is not None:
        validate_policy(policy)
    contract = source_contract() if contract is None else contract
    defaults = contract['default_caps']
    policies = {'frozen_defaults': defaults}
    if policy is not None:
        policies['supplied_policy'] = {key: policy.get(key, defaults[key]) for key in CAPS}
    original_pairs = n * (n - 1) // 2 * c
    retained_pairs = r * (r - 1) // 2 * c
    scenarios = []
    for q in outputs:
        packed = d * (d + 1) // 2 + d * q + 1
        selected = 0 if r == 0 and selected_records is None else selected_records
        keys = 0 if r == 0 and symbolic_keys is None else symbolic_keys
        comparisons = {}
        for name, caps in policies.items():
            pair = compare_count(original_pairs, original_pairs, original_pairs, caps[PAIR_CAP], 'pair-coordinate cap units')
            pair.update(maximum_original_records_at_this_curator_dimension=maximum_pair_rows(caps[PAIR_CAP], c),
                        retained_checkpoint_demand=retained_pairs,
                        executed_for='Actual FP64 state-audit calls: B-E/P-I/P-S/P-R light audits; all eight methods when full-state flagged.',
                        scope='Original graph is checked before the audit service; each retained graph is checked separately.')
            coefficient = compare_count(None if keys is None else keys * packed, 0 if keys is None else keys * packed,
                2 * r * packed if keys is None else keys * packed, caps[COEFFICIENT_CAP], 'designated-symbol-coordinate cap units')
            coefficient.update(maximum_designated_keys=caps[COEFFICIENT_CAP] // packed,
                               executed_for='Full-state P-I, P-S, P-R checkpoints only',
                               applied_in_this_scope=audit_scope == 'full')
            if audit_scope == 'light':
                coefficient['status'] = 'not_applied_in_light_audit'
            convex = compare_count(None if selected is None else selected * d * q,
                0 if selected is None else selected * d * q,
                r * d * q if selected is None else selected * d * q,
                caps[CONVEX_CAP], 'output-row-coordinate cap units')
            convex.update(maximum_selected_records=caps[CONVEX_CAP] // (d * q),
                          all_retained_selected_bound=r * d * q,
                          executed_for='Convex verifier calls on their actual selected training rows.',
                          backend_scope='Both fraction_reference and exact_dyadic_integer_v1 use this same count gate.')
            gates = [pair, convex] + ([coefficient] if audit_scope == 'full' else [])
            status = ('known_count_refusal' if any(g['status'] == 'refused_by_count_cap' for g in gates)
                      else 'unknown_counts_remain' if any(g['status'] == 'unknown_count_required' for g in gates)
                      else 'count_caps_only_pass_resource_qualification_absent')
            comparisons[name] = {'status': status,
                                 'status_scope': 'combined_gate_inventory_not_method_or_job_admission',
                                 'applicability': 'These gates serve different job branches. A convex-only job need not execute the state audit.',
                                 'graph_audit': pair, 'full_coefficients': coefficient,
                                 'convex_verification': convex}
        scenarios.append({'outputs': q, 'packed_coordinates_per_designated_symbol': packed,
                          'comparisons': comparisons})
    return {'schema': SCHEMA, 'evidence_role': 'source_bound_integer_arithmetic_not_an_experiment',
            'source_contract': contract,
            'inputs': {'original_records': n, 'learner_dimension': d, 'curator_dimension': c,
                'curator_dimension_equals_learner_by_default': curator_dimension is None,
                'retained_records': r, 'retained_records_default_to_original': retained_records is None,
                'selected_records': selected_records, 'designated_symbolic_keys': symbolic_keys,
                'audit_scope': audit_scope, 'outputs': list(outputs)},
            'supplied_policy': policy, 'scenarios': scenarios,
            'unknowns': ['Actual retained selected count, unless supplied or the panel is empty.',
                'Designated symbolic key count at the remaining horizon, unless supplied or empty.',
                'Stored and actual/fresh union key counts, graph edges, and incidence multiplicities.',
                'All other checkpoints and repeated audit services; no trajectory total is inferred.',
                'Runtime, peak RSS, graph/index storage, serialization, integer growth, and solver work.'],
            'qualification': {'runtime_seconds': None, 'peak_rss_bytes': None,
                'observed_native_run': False, 'measured_timeout_or_oom': False,
                'policy_approved': False, 'development_lock_issued': False,
                'primary_input_acceptance': False,
                'required_before_execution': [
                    'Genuine accepted corpus, encoder, calibration, and source inputs under the original study gates.',
                    'Prospective reviewed audit caps and common resource policy, bound before confirmation.',
                    'Actual suitable development measurements and the existing convex budget selector for larger verifier caps.',
                    'Dense verifier observations matching learner dimension and output count; source hashes and input bytes remain bound.',
                    'Runtime and peak-memory qualification remain separate from count-cap admission.'],
                'limits': [
                    'Pair cap counts unique undirected pair coordinates; tiled scoring also evaluates masked or diagonal entries.',
                    'The coefficient cap limits fresh symbolic coordinates, not all temporary arrays or union comparison work.',
                    'Each coefficient can sum many record statistics; coordinate count does not measure this accumulation work.',
                    'Light audits still rebuild the original and each retained graph; they do not bypass the pair cap.',
                    'Larger counts do not relax release tolerance, sigmoid limits, state equality, or authenticity requirements.']}}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--records', type=int, required=True, help='Original raw panel record count, before any deletion.')
    result.add_argument('--dimension', type=int, required=True, help='Learner feature dimension.')
    result.add_argument('--outputs', type=int, nargs='+', required=True, help='One or more output counts, evaluated separately.')
    result.add_argument('--curator-dimension', type=int, help='Graph feature dimension; defaults explicitly to learner dimension.')
    result.add_argument('--retained-records', type=int, help='One checkpoint count; defaults to original count, without claiming a deletion path.')
    result.add_argument('--selected-records', type=int, help='Actual selected count at that checkpoint; omission preserves uncertainty.')
    result.add_argument('--symbolic-keys', type=int, help='Actual len(fresh_symbols) at the remaining horizon; not stored basis size.')
    result.add_argument('--audit-scope', choices=('full', 'light'), default='full')
    result.add_argument('--policy', type=Path, help='Optional complete explicit dispatcher policy, compared but not approved.')
    result.add_argument('--out', type=Path, help='New report file; existing files are never overwritten. Omit for stdout only.')
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.out is not None and (args.out.exists() or args.out.is_symlink()):
            raise FileExistsError('Preserve existing output: ' + str(args.out))
        raw_policy = None if args.policy is None else args.policy.read_bytes()
        policy = None if raw_policy is None else loads(raw_policy.decode('utf-8'))
        report = preflight(records=args.records, dimension=args.dimension, outputs=args.outputs,
            curator_dimension=args.curator_dimension, retained_records=args.retained_records,
            selected_records=args.selected_records, symbolic_keys=args.symbolic_keys,
            audit_scope=args.audit_scope, policy=policy)
        report['preflight_source_sha256'] = sha256(Path(__file__).read_bytes())
        report['policy_file_sha256'] = None if raw_policy is None else sha256(raw_policy)
        rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.out is not None:
            with args.out.open('x') as stream:
                stream.write(rendered)
        print(rendered, end='')
        return report
    except (ValueError, TypeError, OSError, SyntaxError) as error:
        p.exit(2, 'BLOCKED: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
