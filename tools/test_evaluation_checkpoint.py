"""Evaluation prefix integrity and completion recovery regressions without CUDA."""
import json
from pathlib import Path
import tempfile
import unittest
from evaluation_checkpoint import atomic_json, fingerprint, prepare, recover_prefix, summarize


class EvaluationCheckpointTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'runtime/tmp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='eval-resume-',dir=base);self.root=Path(self.temp.name)
        self.output=self.root/'result.json'
        self.selected=[dict(sample_id='a'),dict(sample_id='b')]
        self.identity=dict(model='pinned-fixture',csv='pinned-source')

    def tearDown(self):self.temp.cleanup()

    @staticmethod
    def verify(row,source):
        if row['sample_id']!=source['sample_id']:raise ValueError('Not an exact prefix')

    def test_durable_prefix_torn_tail_and_completion_are_preserved(self):
        rows,done=prepare(self.output,self.identity,self.selected,self.verify)
        self.assertEqual(rows,[]);self.assertIsNone(done)
        journal=self.output.with_suffix('.jsonl')
        a=dict(sample_id='a',category='CW',correct=True,invalid=False)
        journal.write_bytes((json.dumps(a)+'\n').encode()+b'{"sample')
        rows,done=prepare(self.output,self.identity,self.selected,self.verify)
        self.assertEqual(rows,[a]);self.assertTrue(journal.read_bytes().endswith(b'\n'))
        b=dict(sample_id='b',category='TD',correct=False,invalid=True)
        with journal.open('a') as stream:stream.write(json.dumps(b)+'\n')
        atomic_json(self.output,dict(summary=summarize([a,b]),res_list=[a,b]))
        rows,done=prepare(self.output,self.identity,self.selected,self.verify)
        self.assertEqual(rows,[a,b]);self.assertEqual(done['summary']['total'],2)

    def test_midstream_corruption_and_changed_inputs_are_rejected(self):
        prepare(self.output,self.identity,self.selected,self.verify)
        journal=self.output.with_suffix('.jsonl');journal.write_text('broken\n{}\n')
        with self.assertRaises(ValueError):prepare(self.output,self.identity,self.selected,self.verify)
        journal.write_text('')
        with self.assertRaisesRegex(ValueError,'changed'):prepare(self.output,dict(model='different'),self.selected,self.verify)

    def test_wrong_order_and_inconsistent_completed_metrics_are_rejected(self):
        prepare(self.output,self.identity,self.selected,self.verify)
        journal=self.output.with_suffix('.jsonl')
        journal.write_text(json.dumps(dict(sample_id='b'))+'\n')
        with self.assertRaises(ValueError):prepare(self.output,self.identity,self.selected,self.verify)
        rows=[dict(sample_id=x,category='CW',correct=True,invalid=False) for x in ['a','b']]
        journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        atomic_json(self.output,dict(summary=dict(total=2,correct=2,accuracy=0),res_list=rows))
        with self.assertRaisesRegex(ValueError,'summary'):prepare(self.output,self.identity,self.selected,self.verify)

    def test_actual_weight_replacement_changes_model_fingerprint(self):
        model=self.root/'model';model.mkdir();weight=model/'model.safetensors';weight.write_bytes(b'fixture-model-a')
        (model/'config.json').write_text('{}');csv=self.root/'val.csv';csv.write_text('fixture')
        first=fingerprint(model,csv,self.root,__file__,'val',2,'sdpa')
        (model/'training_metrics.json').write_text('{"final_export_recovered":true}')
        self.assertEqual(first,fingerprint(model,csv,self.root,__file__,'val',2,'sdpa'))
        weight.write_bytes(b'fixture-model-b')
        second=fingerprint(model,csv,self.root,__file__,'val',2,'sdpa')
        self.assertNotEqual(first,second)


if __name__=='__main__':unittest.main()
