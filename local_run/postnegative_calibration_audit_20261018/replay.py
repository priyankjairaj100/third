#!/usr/bin/env python3
"""Read-only forensic replay; no inference, source-text reads or gate changes."""
import hashlib, importlib.util, json
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
ROOT=Path('/Users/priyankjairaj/Downloads/ACL/Three')
OUT=Path(__file__).resolve().parent
PRIVATE=ROOT/'local_workspace/reports/llm-v4-independent-selection-review-a01'
source=ROOT/'local_run/llm_v4_negative_analysis_v1_20261016/analyze.py'
spec=importlib.util.spec_from_file_location('existing_negative_replay',source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
verified=mod.analyze(PRIVATE)
rows=[json.loads(x) for x in (PRIVATE/'sample.private.jsonl').read_bytes().splitlines()]
audits={m:json.loads((PRIVATE/(m+'.private.json')).read_bytes()) for m in ('qwen','mistral')}
labels={m:{x['opaque_pair_id']:x for x in a['canonical_observations']} for m,a in audits.items()}
pos=lambda x:x['status']=='valid' and x['category'] in ('exact_copy','substantially_same_meaning')
weight=lambda r:Fraction(r['inclusion_weight']['numerator'],r['inclusion_weight']['denominator'])
f=lambda x:{'numerator':x.numerator,'denominator':x.denominator}
cut_diagnostics=[]
for tau in mod.GRID:
 subset=[r for r in rows if float.fromhex(r['score_hex'])>float(tau)]
 den=sum((weight(r) for r in subset),Fraction())
 rates={m:sum((weight(r)*pos(labels[m][r['opaque_pair_id']]) for r in subset),Fraction())/den for m in labels}
 # Only invalid/missing statuses receive favorable imputation, distinct from uncertain labels.
 fmt= sum((weight(r)*all(pos(labels[m][r['opaque_pair_id']]) or labels[m][r['opaque_pair_id']]['status']!='valid' for m in labels) for r in subset),Fraction())/den
 cut_diagnostics.append({'tau':f(tau),'single_model_positive_rates':{m:f(v) for m,v in rates.items()},'joint_format_only_favorable_imputation':f(fmt)})
rat=lambda v:Fraction(v['numerator'],v['denominator'])
single_max={m:max((rat(x['single_model_positive_rates'][m]) for x in cut_diagnostics)) for m in labels}
files=[
'local_run/model_annotation_v4_20261014/PLAN.json',
'local_run/model_annotation_v4_20261014/sampling.py',
'local_run/model_annotation_v4_20261014/select_worker.py',
'local_run/model_annotation_v2_20261011/judge_system_prompt.txt',
'local_run/model_annotation_v2_20261011/judge_task_definition.txt',
'local_run/model_annotation_v2_20261011/model_judgment.schema.json',
'local_run/model_annotation_runtime_v5_20261011/run_pilot.py',
'local_run/model_annotation_runtime_mistral_v1_20261011/run_pilot.py',
'local_run/model_annotation_runtime_v5_20261011/bounded_reason.py',
'local_run/llm_v4_inference_v2_20261014/kernel.py',
'local_run/llm_v4_inference_v2_20261014/statistics_core.py',
'local_run/llm_v4_audit_repair_v1_20261015/audit.py',
'local_run/llm_v4_audit_repair_v1_20261015/DERIVATION.json',
'local_run/llm_v4_negative_analysis_v1_20261016/analyze.py',
'local_run/local_llm_execution_20261014/llm_v4_negative_diagnostics_20261016.json',
]
report={
 'schema':'third-v4-calibration-forensic-review-20261018',
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Independent AI source inspection and read-only replay of existing private label aggregates. No inference or semantic reannotation.',
 'new_model_calls':0,'human_annotation_calls':0,'remote_attempts':0,'repo_mutations':0,
 'conclusion':'No confirmed implementation defect explaining or reversing v4 calibration failure was found in this bounded review. Preserve terminal negative v4. New experimental continuation requires explicit versioning and disclosed exposure; the app UI runtime is separate.',
 'current_exact_51_cut_replay_passed':verified['all51_full_cut_records_exactly_recomputed'],
 'selection_passed':False,'selected_tau':None,'selection_requirement':{'numerator':19,'denominator':20},
 'joint_max':verified['maximum_weighted_joint_positive_proportion'],
 'joint_missing_and_uncertain_favorable_max':verified['maximum_missing_uncertain_sensitivity'],
 'joint_format_only_favorable_max':f(max(rat(x['joint_format_only_favorable_imputation']) for x in cut_diagnostics)),
 'single_model_maxima_descriptive_not_alternate_gates':{m:f(v) for m,v in single_max.items()},
 'raw_replay_failure_counts':{m:dict(Counter(json.dumps(r['independent_failure_checks'],sort_keys=True) for r in a['raw_replay_results'])) for m,a in audits.items()},
 'findings':[
  {'id':'F1','kind':'verified_existing_defect_already_repaired','finding':'The original CPU auditor normalized CR/CRLF in six stored outputs; the separate exact-byte repair retained all labels and model outputs and passed. It was not a generator or scoring fault.', 'evidence':['local_run/llm_v4_audit_repair_v1_20261015/DERIVATION.json','local_run/llm_v4_audit_repair_v1_20261015/audit.py']},
  {'id':'F2','kind':'source_and_receipt_check','finding':'Both renderers supply the same fixed system/task rubric and complete A/B original texts using pinned native chat templates; no score, expected label, metadata or opposite-model answer is supplied. CPU audits independently reconstructed every prompt/token sequence and pair envelope.', 'evidence':['local_run/model_annotation_runtime_v5_20261011/run_pilot.py:553','local_run/model_annotation_runtime_mistral_v1_20261011/run_pilot.py:571','local_run/llm_v4_audit_repair_v1_20261015/audit.py:77']},
  {'id':'F3','kind':'formatting_not_sufficient_explanation','finding':'Only five Qwen calls failed strict JSON parsing and no Mistral call failed. Favorable unavailable-only and unavailable-plus-uncertain imputation remain below the gate; formatting fixes cannot admit this result.', 'evidence':['local_run/llm_v4_negative_analysis_v1_20261016/analyze.py','private audit bindings retained below']},
  {'id':'F4','kind':'scoring_source_review','finding':'The implementation uses strict score > float(tau), exact Fraction inverse-inclusion weights, full denominators and the two-valid-positive intersection required by the frozen plan. All 51 terminal cuts independently reproduce.', 'evidence':['local_run/llm_v4_inference_v2_20261014/statistics_core.py:122','local_run/model_annotation_v4_20261014/sampling.py:11']},
  {'id':'F5','kind':'rater_limitation_not_bug_proof','finding':'Substantial model disagreement remains, with model-specific maximum positive rates 1/2 and 5/6. Neither alone reaches 19/20 on the existing grid. These descriptive rates do not authorize selecting a single rater or changing the target.', 'evidence':['cut_diagnostics in this report']},
  {'id':'F6','kind':'remaining_uncertainty','finding':'The custom grammar/word/standalone-token filter is intentionally narrower than unconstrained native greedy decoding; code verifies prefix consistency and records rejections. No independent full-logit replay or matched unconstrained comparison is available. This is a disclosed behavior and possible future engineering diagnostic, not a confirmed bug or evidence that valid labels are wrong.', 'evidence':['local_run/model_annotation_runtime_v5_20261011/bounded_reason.py:245','local_run/llm_v4_inference_v2_20261014/kernel.py:140']},
 ],
 'safe_follow_on':[
  'Preserve v3/v4 outcomes and original unmet gates. Do not restart the completed selection controller, promote failed artifacts, or run blocked validation/guard/effect stages unchanged.',
  'If remote access is restored, retrieve only already-existing private raw responses, request mappings, output tokens and guard-rejection ledgers into a new private diagnostic root. Reconcile scheduler first; this audit made no remote attempt.',
  'A new, prospectively frozen engineering study can test exact algorithm/oracle agreement and resource costs on natural development data using an explicitly fixed graph; it cannot be called the gated semantic effect study or semantic-quality evidence.',
  'A substantively different model/representation/prompt/label definition requires a new protocol and evaluation with v3/v4 exposure disclosed; retain an untouched fresh validation stage and do not inspect held-out effect outcomes for tuning.',
  'A narrowly scoped decoder diagnostic may compare current versus alternative decoding on already exposed engineering pairs with fixed decisions and no relabeling of v4. Qualify the runtime and record all calls, including failures, before any substantive new evaluation.'
 ],
 'limits':['No semantic source-text rereview or new label judgments were performed.','Saved independent audits were authenticated by fixed file hashes and seals but model weights were not rehashed locally.','No full calibration pair census or raw model logit replay was rerun.','The native-app PDF runtime failure cannot explain these independent completed Slurm results.'],
 'private_input_bindings':mod.PINS,
 'source_bindings':{str(p):{'bytes':(ROOT/p).stat().st_size,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in files},
 'cut_diagnostics':cut_diagnostics,
 'publication_policy':'Aggregate diagnostics only. No private source text, pair/record IDs, reasons or per-pair labels.'
}
(OUT/'diagnostic.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:report[k] for k in ('conclusion','current_exact_51_cut_replay_passed','joint_max','joint_format_only_favorable_max','joint_missing_and_uncertain_favorable_max','single_model_maxima_descriptive_not_alternate_gates')},indent=2))
