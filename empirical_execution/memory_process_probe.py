#!/usr/bin/env python3
"""One fresh-process peak-RSS observation; never used as a speed benchmark."""
import argparse,json,resource,time
from pathlib import Path
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features
from ccu.core import build_blocker_graph
from ccu.summary import IndexedRidgeSummary
from ccu.factored_summary import FactoredRidgeSummary
from ccu.joint_span_summary import JointSpanRidgeSummary
from ccu.compact_payload import CompactEligiblePayloadState

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('method',choices=['dense','gram_factor','joint_span','compact_payload'])
args=parser.parse_args()
rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)
z,_=lexical_engineering_features(rows,768);e,_=lexical_engineering_features(rows,128)
y=np.asarray([r['label'] for r in rows]);ids=[r['record_id'] for r in rows]
g=build_blocker_graph(e,ids,.6)
cls={'dense':IndexedRidgeSummary,'gram_factor':FactoredRidgeSummary,
     'joint_span':JointSpanRidgeSummary,'compact_payload':CompactEligiblePayloadState}[args.method]
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
begin=time.perf_counter();s=cls.build(z,y,g.blockers,8)
after_build=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
construction=time.perf_counter()-begin
initial=s.accounting();max_residual=0.
for request in [[0],[1],[2,3],[4,5,6,7]]:
    s.delete(request)
    solution=s.decode_compact(.01) if args.method=='joint_span' else s.decode(.01)
    max_residual=max(max_residual,solution.normal_equation_residual_fro)
report={'method':args.method,'dimension':768,'outputs':1,'initial_accounting':initial,
        'peak_RSS_KiB_after_setup':before,'peak_RSS_KiB_after_build':after_build,
        'peak_RSS_KiB_after_four_releases':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'build_seconds_unreplicated':construction,'max_normal_equation_residual':max_residual,
        'scope':'whole fresh process, one observation, shared original input/graph/runtime included; no oracle arrays or audit reconstructions',
        'decoder':'compact reduced solve' if args.method=='joint_span' else 'method default',
        'not_per_method_attributable_bytes':True}
print(json.dumps(report))
