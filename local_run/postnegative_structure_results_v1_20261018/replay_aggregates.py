#!/usr/bin/env python3
"""Replay complete Stage1 aggregates with Python 3.9+ standard library only.

Does not access the network, private inputs, an external checkout or GPUs.
This is arithmetic/provenance replay, not graph or semantic validation.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

if sys.flags.optimize:
    raise SystemExit("Run without -O: assertions implement audit checks.")
ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'

def binding(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result, 'Duplicate JSON key'
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=nonfinite)

manifest = read(ROOT / 'MANIFEST.json')
assert manifest['schema'] == 'third-stage1-public-bundle-manifest-v1'
assert not any(p.is_symlink() for p in ROOT.rglob('*'))
actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
assert actual == set(manifest['files']) | {'MANIFEST.json'}, 'Unexpected/missing file'
for name, expected in manifest['files'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert binding(ROOT / name) == expected, 'Hash mismatch: ' + name
expected_runs = {'completion.json', 'intake.json', 'subset_oracle.json'} | {
    'threshold_%02d.json' % i for i in range(51)}
assert {p.name for p in RUN.iterdir()} == expected_runs
completion = read(RUN / 'completion.json')
intake = read(RUN / 'intake.json')
subset = read(RUN / 'subset_oracle.json')
plan = read(ROOT / 'study_protocol.json')
resources = read(ROOT / 'resource_receipt.json')
sources = read(ROOT / 'SOURCE_BINDINGS.json')
review = read(ROOT / 'independent_review.json')
assert review['passed'] is True
assert completion['status'] == 'completed' and completion['all_thresholds_completed'] is True
assert set(completion['outputs']) == expected_runs - {'completion.json'}
for name, expected in completion['outputs'].items():
    assert binding(RUN / name) == expected
for key, name in (('completion', 'completion.json'), ('intake', 'intake.json'),
                  ('subset_oracle', 'subset_oracle.json')):
    assert review['bindings'][key] == binding(RUN / name)
assert (sources['executed_protocol'] == review['bindings']['protocol'] ==
        plan['public_transformation']['original_protocol_binding'] ==
        resources['protocol_binding'] == sources['executed_package_members']['PLAN.json'])
assert intake['protocol_sha256'] == sources['executed_protocol']['sha256']
assert (sources['executed_package_manifest'] == review['bindings']['package_manifest'] ==
        resources['package_manifest_binding'])
assert plan['public_transformation']['executed_document'] is False
assert sources['frozen_research_sources'] == plan['frozen_sources']
assert intake['runtime'] == plan['runtime'] == resources['runtime']
assert intake['runtime'] == {'platform': 'Linux', 'python': '3.13.7',
                             'numpy': '2.5.3', 'scipy': '1.17.1'}
assert (intake['local_inputs'] == intake['original_inputs'] == plan['inputs'] ==
        sources['private_original_inputs'])
assert intake['rows'] == 1001 and intake['feature_slice_roles'] == ['development_train']
assert all(intake[key] is False for key in ('heldout_feature_slices_read', 'labels_read', 'text_read'))
assert all(completion[key] is False for key in ('primary_study_started', 'repair_effects_measured', 'semantic_threshold_selected'))
assert completion['human_reviews'] == completion['model_calls'] == 0
assert completion['original_v4_failure_retained'] is True
assert subset['passed'] is True and subset['rows'] == 16 and subset['thresholds'] == 51
assert subset['blocker_set_comparisons'] == 816
assert resources['process_returncode'] == 0 and resources['private_feature_or_label_output'] is False
assert resources['scratch_free_bytes_at_exit'] >= resources['scratch_reserve_bytes'] == 8 * 1024**3
assert resources['cpus_requested'] == 2 and resources['gpus'] == 0
assert resources['memory_requested_bytes'] == 8 * 1024**3
assert resources['stage_seconds'] == completion['elapsed_seconds']
assert resources['peak_rss_bytes'] == completion['peak_rss_bytes'] < resources['memory_requested_bytes']
assert 0 < resources['stage_seconds'] <= resources['wrapper_seconds'] < resources['stage_wall_limit_seconds'] == 1200
assert plan['horizons'] == [1, 8, 32, 128]
assert plan['thresholds'] == [[i,100] for i in range(50,100)] + [[199,200]]
grid=[Fraction(i,100) for i in range(50,100)]+[Fraction(199,200)]
rows=[];prior=None;expectation_checks=0;cdf_checks=0
for i,cut in enumerate(grid):
 x=read(RUN/('threshold_%02d.json'%i));n=x['rows'];assert n==1001
 assert x['index']==i and x['threshold_fraction']==[cut.numerator,cut.denominator]
 assert x['threshold']==float(cut) and x['threshold_float_hex']==float(cut).hex()
 assert x['semantic_validity_claim'] is False and x['primary_study_result'] is False
 assert x['full_graph_alternate_tiling_equal'] is True
 hist={int(k):v for k,v in x['blocker_histogram'].items()}
 assert len(hist)==len(x['blocker_histogram']) and all(type(v) is int and v>0 and 0<=b<n for b,v in hist.items())
 assert sum(hist.values())==n and sum(b*v for b,v in hist.items())==x['edge_count']
 assert x['initially_selected']==hist.get(0,0) and x['initially_excluded']==n-hist.get(0,0)
 assert 0<=x['edge_count']<=n*(n-1)//2
 assert x['singleton_record_admission_total']==hist.get(1,0)
 assert 0<=x['singleton_excluded_blocker_admission_total']<=x['singleton_record_admission_total']
 assert 0<=x['singleton_record_requests_with_admission']<=x['singleton_record_admission_total']
 assert 0<=x['singleton_excluded_record_requests_with_admission']<=min(x['singleton_excluded_blocker_admission_total'],x['singleton_record_requests_with_admission'],x['initially_excluded'])
 assert x['genuine_single_source_request_count']==457
 assert 0<=x['genuine_single_source_requests_with_admission']<=min(457,x['genuine_single_source_admission_total'])
 assert 0<=x['genuine_single_source_admission_total']<=x['initially_excluded']
 assert x['array_and_utf8_accounting']=={'records':n,'edges':x['edge_count'],'array_bytes':8*(2*n+1+x['edge_count']),'id_utf8_bytes':39173}
 for k in (1,8,32,128):
  assert x['eligible_record_count_by_horizon'][str(k)]==sum(v for b,v in hist.items() if b<=k)
  # Independently use falling factorials and candidate survival, instead of binomial coefficient ratio.
  value=Fraction(0)
  for b,count in hist.items():
   if b==0 or b>k:continue
   probability=Fraction(n-k,n-b)
   for j in range(b):probability*=Fraction(k-j,n-j)
   value+=count*probability
  z=x['uniform_record_expected_admissions'][str(k)]
  assert z['deletion_budget']==k and Fraction(int(z['numerator']),int(z['denominator']))==value
  assert z['numerator']==str(value.numerator) and z['denominator']==str(value.denominator)
  assert math.isfinite(z['float_display']) and z['float_display']==float(value)
  expectation_checks+=1
 if prior is not None:
  assert x['edge_count']<=prior['edge_count'] and x['initially_selected']>=prior['initially_selected']
  before={int(k):v for k,v in prior['blocker_histogram'].items()}
  cdf=oldcdf=0
  for b in range(n):
   cdf+=hist.get(b,0);oldcdf+=before.get(b,0);assert cdf>=oldcdf;cdf_checks+=1
 prior=x
 assert math.isfinite(x['graph_and_replay_seconds']) and x['graph_and_replay_seconds']>0
 rows.append({k:x[k] for k in ('index','threshold_fraction','edge_count','initially_selected','initially_excluded','eligible_record_count_by_horizon','singleton_record_admission_total','singleton_excluded_blocker_admission_total','singleton_record_requests_with_admission','singleton_excluded_record_requests_with_admission','genuine_single_source_admission_total','genuine_single_source_requests_with_admission','uniform_record_expected_admissions','graph_and_replay_seconds')})
assert sum(x['graph_and_replay_seconds'] for x in rows)<=completion['elapsed_seconds']
# A positive result means structurally possible admission only, not semantic quality.
summary={'rows':1001,'thresholds':51,'complete_graph_cut_count':sum(x['edge_count']==500500 for x in rows),'empty_graph_cut_count':sum(x['edge_count']==0 for x in rows),'cuts_with_any_singleton_record_admission':sum(x['singleton_record_admission_total']>0 for x in rows),'cuts_with_excluded_record_deletion_admission':sum(x['singleton_excluded_blocker_admission_total']>0 for x in rows),'cuts_with_any_genuine_source_admission':sum(x['genuine_single_source_admission_total']>0 for x in rows),'threshold_selected':False,'human_or_model_annotations':0}

assert summary == review['summary']
assert rows == review['full51cut_summary']
assert expectation_checks == review['checks']['exact_expectation_checks'] == 204
assert cdf_checks == review['checks']['full_blocker_cdf_monotonicity_checks'] == 50050
assert sum(x['graph_and_replay_seconds'] for x in rows) == resources['graph_and_replay_seconds_sum']
for key in ('stage_seconds', 'wrapper_seconds', 'peak_rss_bytes', 'runtime',
            'gpus', 'cpus_requested', 'graph_and_replay_seconds_sum'):
    assert review['runtime_summary'][key] == resources[key]
assert review['original_v4_failure_retained'] is True
assert review['original_semantic_effect_study_complete'] is False
print(json.dumps({
    'passed': True,
    'bundle_files_checked': len(manifest['files']),
    'completion_output_hashes_checked': len(completion['outputs']),
    'exact_expectations_replayed': expectation_checks,
    'blocker_cdf_comparisons': cdf_checks,
    'all51cuts': True,
    'summary': summary,
    'scope': 'Portable aggregate replay only; private graphs and original execution metadata were not re-created.'
}, indent=2, sort_keys=True))
