"""Scoped device discovery and concurrent evaluator failure isolation."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from keepalive_guard import gpu_pids
from parallel_evaluations import run_jobs

class ResourceTests(unittest.TestCase):
    def test_gpu_scope_excludes_foreign_group(self):
        devices='0, U0\n1, U1\n4, U4\n7, U7\n'
        processes='U0, 10\nU1, 11\nU4, 40\nU7, 70\n'
        with patch('keepalive_guard.subprocess.check_output',side_effect=[devices,processes]):
            self.assertEqual(gpu_pids(['0','1']),{10,11})
        with patch('keepalive_guard.subprocess.check_output',side_effect=[devices,processes]):
            self.assertEqual(gpu_pids(['4','7']),{40,70})
    def test_four_eval_jobs_are_assigned_distinct_devices(self):
        folder=Path(__file__).resolve().parents[1]/'runtime/tmp';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            jobs=[dict(tag=str(i),command=[sys.executable,'-B','-c',
                 'import os,pathlib,time;pathlib.Path('+repr(str(Path(temp)/str(i)))+').write_text(os.environ["CUDA_VISIBLE_DEVICES"]);time.sleep(.5)']) for i in range(4)]
            self.assertEqual(set(run_jobs(jobs,['0','1','2','3'])),{'0','1','2','3'})
            self.assertEqual({(Path(temp)/str(i)).read_text() for i in range(4)},{'0','1','2','3'})
    def test_failure_terminates_other_evaluation(self):
        folder=Path(__file__).resolve().parents[1]/'runtime/tmp';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            pidfile=Path(temp)/'pid'
            code='import os,pathlib,time;pathlib.Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(60)'
            jobs=[dict(tag='slow',command=[sys.executable,'-B','-c',code]),
                  dict(tag='bad',command=[sys.executable,'-B','-c','import time;time.sleep(.5);raise SystemExit(7)'])]
            with self.assertRaises(RuntimeError):run_jobs(jobs,['0','1'])
            self.assertFalse(Path('/proc/'+pidfile.read_text()).exists())
if __name__=='__main__':unittest.main()
