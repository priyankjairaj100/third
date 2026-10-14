"""Adversarial review tests using disposable copies of the actual diagnostic evidence."""
import argparse
import base64
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_review',HERE/'audit_qualification.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
REPO=Path('/Users/priyankjairaj/Downloads/ACL/Three')
ARTIFACTS=REPO/'local_workspace/reports/llm-selection-sampler-v1-20261014/qualification-a01'

class Adversarial(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()/'artifacts';shutil.copytree(ARTIFACTS,self.root)
        self.args=argparse.Namespace(repo=REPO,artifacts=self.root,accounting_json=self.root/'terminal_export.json')
    def reseal(self, altered=None):
        output=self.root/'run/output';result=audit.strict((output/'result.json').read_bytes())
        if altered:
            result['files'][altered]=audit.binding((output/altered).read_bytes())
        result['sha256']=audit.binding(audit.canonical({k:v for k,v in result.items() if k!='sha256'}))['sha256']
        (output/'result.json').write_bytes(audit.canonical(result)+b'\n')
        export=audit.strict(self.args.accounting_json.read_bytes())
        for name,entry in export['files'].items():
            raw=(self.root/name).read_bytes();export['files'][name]={**audit.binding(raw),'base64':base64.b64encode(raw).decode()}
        export['total_bytes']=sum(x['bytes'] for x in export['files'].values())
        self.args.accounting_json.write_bytes(audit.canonical(export)+b'\n')
    def test_changed_transported_file_refused(self):
        with (self.root/'run/tests.txt').open('ab') as stream:stream.write(b'changed\n')
        with self.assertRaisesRegex(ValueError,'transport_binding'):audit.validate(self.args)
    def test_fully_resealed_wrong_score_stream_refused(self):
        path=self.root/'run/output/result.json';result=audit.strict(path.read_bytes());result['score_stream_sha256']='0'*64
        path.write_bytes(audit.canonical(result));self.reseal()
        with self.assertRaisesRegex(ValueError,'full512_score_stream'):audit.validate(self.args)
    def test_fully_resealed_wrong_orientation_refused(self):
        path=self.root/'run/output/selection.private.jsonl';rows=[audit.strict(line) for line in path.read_bytes().splitlines()]
        rows[0]['canonical_frame_row_indices'].reverse()
        path.write_bytes(b''.join(audit.canonical(row)+b'\n' for row in rows));self.reseal(path.name)
        with self.assertRaisesRegex(ValueError,'reservoir_identity_weights'):audit.validate(self.args)
    def test_fully_resealed_wrong_reservoir_member_refused(self):
        path=self.root/'run/output/selection.private.jsonl';rows=[audit.strict(line) for line in path.read_bytes().splitlines()]
        rows[0]=dict(rows[1])
        path.write_bytes(b''.join(audit.canonical(row)+b'\n' for row in rows));self.reseal(path.name)
        with self.assertRaisesRegex(ValueError,'reservoir_identity_weights'):audit.validate(self.args)

if __name__=='__main__':unittest.main()
