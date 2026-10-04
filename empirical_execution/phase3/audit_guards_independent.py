#!/usr/bin/env python3
"""Exhaustive software branch tests for guards; not a corpus experiment."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase3 import panels

def main():
    parts=('train','calibration','test');rank={p:i for i,p in enumerate(parts)}
    checks=[]
    for kind in ('parent','exact','duplicate_link'):
        for left in parts:
            for right in parts:
                rows=[{'record_id':'software-left','text':'software field left','partition':left,'original_fields':{}},
                      {'record_id':'software-right','text':'software field right','partition':right,'original_fields':{}}]
                if kind=='parent':rows[0]['original_fields']['parent_id']='software-right'
                if kind=='exact':rows[1]['text']=rows[0]['text']
                links=[('software-left','software-right')] if kind=='duplicate_link' else []
                before=json.dumps(rows,sort_keys=True)
                actual,audit=panels.fixed_guard(rows,links)
                expected={'software-left','software-right'}
                if kind=='parent':
                    if left=='train' and right in ('calibration','test'):expected.remove('software-left')
                elif left!=right:
                    expected.remove('software-left' if rank[left]<rank[right] else 'software-right')
                assert {r['record_id'] for r in actual}==expected,(kind,left,right,actual)
                assert json.dumps(rows,sort_keys=True)==before
                assert audit['removed_training_rows']+audit['removed_calibration_rows']==2-len(expected)
                checks.append({'kind':kind,'left_partition':left,'right_partition':right,'retained_ids':sorted(expected)})
    for owner in (-2,-1,0,'-3','0',None,'','   '):
        rid='software-id';source,kind=panels.source_unit({'record_id':rid,'original_fields':{'site':'software-site','OwnerUserId':owner}},'askubuntu')
        assert source=='unknown:'+rid and kind=='unknown_singleton'
    malformed=0
    for value in (True,False,float('nan'),float('inf'),3.4,[],{}):
        try:panels.source_unit({'record_id':'software-id','original_fields':{'publication_id':value,'article_id':1}},'civil_comments')
        except ValueError:malformed+=1
        else:raise AssertionError(('invalid source accepted',value))
    report={'passed':True,'scope':'27 exhaustive metadata/guard branch fixtures; no empirical corpus or invented human labels',
            'guard_branch_checks':len(checks),'checks':checks,'nonpositive_or_missing_account_cases':8,
            'invalid_native_identifier_cases':malformed,
            'code_sha256':hashlib.sha256(Path(panels.__file__).read_bytes()).hexdigest()}
    Path(__file__).with_name('independent_guard_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
