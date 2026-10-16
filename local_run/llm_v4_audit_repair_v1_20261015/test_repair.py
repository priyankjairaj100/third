"""Regression fixtures for exact byte reading and unchanged audit gates."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
import repair_common as R
import derive_sources as D
class RepairTests(unittest.TestCase):
 def fixture(self,raw):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'response.txt';p.write_bytes(raw)
   self.assertEqual(R.read_raw_utf8(p).encode('utf8'),raw)
   self.assertEqual(hashlib.sha256(R.read_raw_utf8(p).encode()).digest(),hashlib.sha256(raw).digest())
   if b'\r' in raw:self.assertNotEqual(p.read_text().encode(),raw)
 def test_crlf_whitespace_remains_exact(self):self.fixture(b'{\r\n"x": 1\r\n}')
 def test_bare_cr_whitespace_remains_exact(self):self.fixture(b'{\r"x": 1\r}')
 def test_lf_whitespace_remains_exact(self):self.fixture(b'{\n"x": 1\n}')
 def test_unicode_and_mixed_lines_remain_exact(self):self.fixture('{\r\n"x":"अé",\r"y":2\n}'.encode('utf8'))
 def test_invalid_utf8_is_rejected_without_replacement(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'raw';p.write_bytes(b'{"x":"\xff"}')
   with self.assertRaises(UnicodeDecodeError):R.read_raw_utf8(p)
 def test_audit_source_changes_are_exactly_declared(self):
  derivation=R.load(R.HERE/'DERIVATION.json')
  for scope in ('selection','validation'):
   base=Path(os.environ.get('THIRD_REPAIR_'+scope.upper()+'_PACKAGE',str(R.W/'tools'/R.EXECUTIONS[scope]['folder'])))
   self.assertEqual(R.binding(base/'audit.py'),derivation['parents'][scope]['audit.py'])
   self.assertEqual(D.audit((base/'audit.py').read_text()),(R.HERE/'audit.py').read_text())
 def test_launch_source_preserves_original_controls(self):
  derivation=R.load(R.HERE/'DERIVATION.json')
  for scope in ('selection','validation'):
   base=Path(os.environ.get('THIRD_REPAIR_'+scope.upper()+'_PACKAGE',str(R.W/'tools'/R.EXECUTIONS[scope]['folder'])))
   self.assertEqual(R.binding(base/'launch.py'),derivation['parents'][scope]['launch.py'])
   self.assertEqual(D.launch((base/'launch.py').read_text()),(R.HERE/'continue_launch.py').read_text())
 def test_foreign_audit_provenance_is_rejected(self):
  report={'passed':True,'package_sha256':R.EXECUTIONS['selection']['pin'],'audit_repair':{'repair_package_sha256':'fixture','scope':'selection','audit_job_id':'1','auditor_source':R.binding(R.HERE/'audit.py')}}
  R.check_report(report,'selection','fixture','1')
  for key,value in [('repair_package_sha256','other'),('scope','validation'),('audit_job_id','2'),('auditor_source',{})]:
   changed=json.loads(json.dumps(report));changed['audit_repair'][key]=value
   with self.assertRaises(ValueError):R.check_report(changed,'selection','fixture','1')
if __name__=='__main__':unittest.main()
