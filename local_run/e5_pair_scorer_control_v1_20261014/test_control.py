"""Bounded local fixtures only: never execute main or real scheduler commands."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import launch_remote as launch
import export_remote as exporter


class ControlTests(unittest.TestCase):
    def test_fixed_11_arg_mapping(self):
        cmd=launch.sbatch_command();index=cmd.index(str(launch.PACKAGE/'benchmark.sbatch'))
        args=cmd[index+1:]
        self.assertEqual(len(args),11)
        self.assertEqual(args[:3],[str(launch.PACKAGE),str(launch.REPO),str(launch.WORKSPACE)])
        self.assertEqual(args[-2:],[str(launch.ROOT/'run'),launch.PACKAGE_SHA])
        self.assertIn('--nodelist=csis.cn1',cmd)

    def test_active_queue_collision(self):
        launch.queue_clear('999|unrelated\n','13500')
        for text in ('13500|other\n','13498_31|task\n','999|third-e5-assembly-13498\n','999|third-e5-scorebench-v1\n'):
            with self.assertRaises(ValueError):launch.queue_clear(text,'13500')

    def test_terminal_requires_success_exact_parent(self):
        row='13500|COMPLETED|0:0|100||csis.cn1\n13500.batch|COMPLETED|0:0|100|1K|csis.cn1\n'
        self.assertEqual(launch.terminal_parent(row,'13500')['State'],'COMPLETED')
        for raw in ('',row+row,row.replace('COMPLETED','RUNNING'),row.replace('0:0','1:0')):
            with self.assertRaises(ValueError):launch.terminal_parent(raw,'13500')
        self.assertEqual(launch.terminal_parent('9|FAILED|1:0|10||csis.cn1','9',False)['State'],'FAILED')

    def test_existing_root_refused(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(launch,'ROOT',Path(tmp)):
            with self.assertRaises(ValueError):launch.refuse_previous()

    def test_submit_once_records_intent_before_call(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(launch,'ROOT',Path(tmp)):
            def submit(*args,**kwargs):
                self.assertEqual(json.loads((Path(tmp)/'launch_intent.json').read_text()),{'reviewed':True})
                return subprocess.CompletedProcess(args[0],0,'13555\n','')
            with mock.patch.object(launch.subprocess,'run',side_effect=submit) as p:
                result=launch.submit_once(['sbatch','example'],{'reviewed':True})
                self.assertEqual(result['job_id'],'13555');self.assertEqual(p.call_count,1)
                with self.assertRaises(FileExistsError):launch.submit_once(['sbatch'],{'reviewed':True})
                self.assertEqual(p.call_count,1)

    def test_unknown_submit_never_retried_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(launch,'ROOT',Path(tmp)):
            with mock.patch.object(launch.subprocess,'run',side_effect=subprocess.TimeoutExpired(['sbatch'],10)) as p:
                with self.assertRaises(subprocess.TimeoutExpired):launch.submit_once(['sbatch'],{'reviewed':True})
                self.assertEqual(p.call_count,1)
            receipt=json.loads((Path(tmp)/'launch_receipt.json').read_text())
            self.assertEqual(receipt['submission_outcome'],'unknown_reconcile_without_retry')
            self.assertTrue((Path(tmp)/'launch_intent.json').exists())

    def test_nonzero_submit_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(launch,'ROOT',Path(tmp)):
            with mock.patch.object(launch.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','quota')) as p:
                with self.assertRaises(ValueError):launch.submit_once(['sbatch'],{})
                self.assertEqual(p.call_count,1)
            self.assertEqual(json.loads((Path(tmp)/'launch_receipt.json').read_text())['returncode'],1)

    def test_read_cap_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file';p.write_bytes(b'abcd')
            with self.assertRaises(ValueError):launch.read_small(p,3)
            q=Path(tmp)/'link';q.symlink_to(p)
            with self.assertRaises(ValueError):launch.read_small(q)
            with self.assertRaises(ValueError):exporter.inventory(Path(tmp))

    def test_export_all_small_files_no_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'run').mkdir();(p/'run/out').write_bytes(b'abc');(p/'empty').write_bytes(b'')
            self.assertEqual(exporter.inventory(p),(['empty','run/out'],3))
            with mock.patch.object(exporter,'TOTAL_CAP',2):
                with self.assertRaises(ValueError):exporter.inventory(p)
            with mock.patch.object(exporter,'MAX_FILES',1):
                with self.assertRaises(ValueError):exporter.inventory(p)

    def test_quota_reserve_age_and_mount(self):
        mount={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'}
        good={'available_soft_bytes':launch.RESERVE+launch.OUTPUT_ALLOWANCE}
        launch.quota_gate(good,120,mount,mount)
        for q,age,m in (({'available_soft_bytes':good['available_soft_bytes']-1},0,mount),
                        (good,120.1,mount),(good,-1,mount),(good,1,{})):
            with self.assertRaises(ValueError):launch.quota_gate(q,age,m,mount)

    def test_benchmark_child_command_mapping(self):
        cmd=launch.benchmark_command()
        self.assertEqual(cmd[:2],[str(launch.WORKSPACE/'envs/encoder-cpu/bin/python'),str(launch.PACKAGE/'benchmark.py')])
        self.assertEqual(cmd[cmd.index('--out')+1],str(launch.ROOT/'run/benchmark'))
        self.assertEqual(cmd[cmd.index('--qualification-result')+1],str(launch.fixed_inputs()['qualification_result']))
        self.assertEqual(len(cmd),22)

    def test_strict_JSON(self):
        for raw in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError):launch.strict(raw)


if __name__=='__main__':unittest.main()
