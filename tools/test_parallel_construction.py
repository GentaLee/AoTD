"""Exercise spawned scheduling, failure propagation and durable resume without CUDA."""
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import build_cot
from construction_io import append_record, recover_journal, should_process


def fake_worker(connection, gpu, output):
    connection.send(('ready', None))
    while True:
        request = connection.recv()
        if request is None: return
        row, trace, timeout = request
        if row.get('crash'): os._exit(7)
        time.sleep(row.get('delay', 0))
        connection.send(('result', dict(row, status=row.get('status','ok'), gpu=gpu)))


class ParallelTests(unittest.TestCase):
    def setUp(self):
        folder=Path(__file__).resolve().parents[1]/'runtime/tmp'
        folder.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=folder)
        self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def run_rows(self,rows):
        return build_cot.parallel_results(rows,self.root,self.root,1,['0','1'])
    @patch.object(build_cot,'construction_worker',fake_worker)
    def test_out_of_order_exactly_once(self):
        rows=[dict(sample_id=str(i),delay=.15 if i==0 else .01) for i in range(10)]
        result=list(self.run_rows(rows))
        self.assertEqual({x['sample_id'] for x in result},{str(i) for i in range(10)})
        self.assertEqual(len(result),10)
        self.assertEqual({x['gpu'] for x in result},{'0','1'})
    @patch.object(build_cot,'construction_worker',fake_worker)
    def test_close_resume_preserves_committed_prefix(self):
        path=self.root/'construct.jsonl';rows=[dict(sample_id=str(i),delay=.01) for i in range(12)]
        iterator=self.run_rows(rows)
        with path.open('a') as stream:
            for i,item in enumerate(iterator):
                append_record(stream,item)
                if i==3: break
        iterator.close();prefix=path.read_bytes();done=recover_journal(path)
        with path.open('a') as stream:
            for item in self.run_rows(x for x in rows if should_process(done.get(x['sample_id']),False)):
                append_record(stream,item)
        self.assertTrue(path.read_bytes().startswith(prefix))
        self.assertEqual(len(path.read_text().splitlines()),12)
        self.assertEqual(len(recover_journal(path)),12)
    @patch.object(build_cot,'construction_worker',fake_worker)
    def test_worker_death_propagates_and_cleans_up(self):
        before={p.pid for p in multiprocessing.active_children()}
        with self.assertRaises(RuntimeError):
            list(self.run_rows([dict(sample_id='a',crash=True),dict(sample_id='b',delay=5)]))
        self.assertEqual({p.pid for p in multiprocessing.active_children()},before)
    @patch.object(build_cot,'construction_worker',fake_worker)
    def test_operational_error_stops_dispatch(self):
        result=list(self.run_rows([dict(sample_id='a',status='error'),dict(sample_id='b',delay=5)]))
        self.assertEqual(result[0]['status'],'error')
        self.assertEqual(len(result),1)
    def test_no_pending_rows_does_not_spawn(self):
        with patch.object(build_cot.multiprocessing,'get_context',side_effect=AssertionError):
            self.assertEqual(list(self.run_rows([])),[])

if __name__=='__main__': unittest.main()
