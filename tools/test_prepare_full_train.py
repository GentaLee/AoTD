"""Interrupted training-pool publication must resume without replacing good files."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from construction_io import records
from prepare_full_train import ARTIFACTS, digest, prepare
from training_protocol import load_protocol


class InterruptedPublication(BaseException):
    pass


class PrepareTrainingTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'runtime/tmp'
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base,prefix='prepare-publication-')
        self.root=Path(self.temp.name)
        media=self.root/'fixture.mp4';media.write_bytes(b'presence-only unit fixture; never enters production')
        protocol=load_protocol()
        for source in protocol['sources']:source['minimum_qa_records']=1
        self.protocol=patch('prepare_full_train.load_protocol',return_value=protocol);self.protocol.start()
        self.qa=self.root/'qa.jsonl';self.cot=self.root/'cot.jsonl'
        qa=[dict(sample_id=source['dataset']+':fixture',dataset=source['dataset'],
                 video_path=str(media),question='Fixture question?',options=['first','second'],answer=1)
            for source in protocol['sources']]
        cot=[dict(sample_id=row['sample_id'],rationale='Fixture explanation only')
             for row,source in zip(qa,protocol['sources']) if source['cot']]
        self.qa.write_text(''.join(json.dumps(row)+'\n' for row in qa))
        self.cot.write_text(''.join(json.dumps(row)+'\n' for row in cot))

    def tearDown(self):
        self.protocol.stop();self.temp.cleanup()

    def interrupt(self, output, after):
        replace=Path.replace;published=[]
        def interrupted(path,target):
            target=Path(target)
            publication=target.parent==output and target.name in ARTIFACTS
            if publication and after==0:raise InterruptedPublication()
            result=replace(path,target)
            if publication:
                published.append(target)
                if len(published)==after:raise InterruptedPublication()
            return result
        with patch.object(Path,'replace',interrupted):
            with self.assertRaises(InterruptedPublication):prepare(self.qa,self.cot,output,True)
        return published

    def test_restart_after_every_publication_boundary_preserves_published_files(self):
        for after in range(len(ARTIFACTS)+1):
            with self.subTest(published_files=after):
                output=self.root/f'train-{after}'
                published=self.interrupt(output,after)
                before={path.name:(digest(path),path.stat().st_mtime_ns) for path in published}
                manifest=prepare(self.qa,self.cot,output,True)
                self.assertEqual((manifest['qa_records'],manifest['cot_records']),(5,3))
                self.assertEqual(len(list(records(output/'answering.json'))),5)
                self.assertEqual(len(list(records(output/'reasoning.json'))),3)
                for name,state in before.items():
                    path=output/name
                    self.assertEqual((digest(path),path.stat().st_mtime_ns),state)
                complete={name:(digest(output/name),(output/name).stat().st_mtime_ns) for name in ARTIFACTS}
                prepare(self.qa,self.cot,output,True)
                self.assertEqual(complete,{name:(digest(output/name),(output/name).stat().st_mtime_ns) for name in ARTIFACTS})

    def test_changed_inputs_during_interruption_are_rejected_without_publication(self):
        output=self.root/'changed';published=self.interrupt(output,1)
        original=published[0].read_bytes()
        self.cot.write_text(self.cot.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed during publication'):
            prepare(self.qa,self.cot,output,True)
        self.assertEqual(published[0].read_bytes(),original)
        self.assertFalse((output/'manifest.json').exists())

    def test_legacy_partial_pool_is_recovered_only_when_existing_content_matches(self):
        output=self.root/'legacy';prepare(self.qa,self.cot,output,True)
        for name in ['publication.json']:(output/'.prepare-publication'/name).unlink()
        (output/'manifest.json').unlink();(output/'aotd.yaml').unlink()
        before=(output/'answering.json').stat().st_mtime_ns
        prepare(self.qa,self.cot,output,True)
        self.assertEqual((output/'answering.json').stat().st_mtime_ns,before)
        self.assertTrue((output/'aotd.yaml').is_file())
        self.assertTrue((output/'manifest.json').is_file())

    def test_foreign_existing_pool_is_never_overwritten(self):
        output=self.root/'foreign';output.mkdir()
        foreign=output/'answering.json';foreign.write_text('preserve unrelated data')
        with self.assertRaisesRegex(ValueError,'Conflicting/incomplete'):
            prepare(self.qa,self.cot,output,True)
        self.assertEqual(foreign.read_text(),'preserve unrelated data')
        self.assertFalse((output/'reasoning.json').exists())
        self.assertFalse((output/'manifest.json').exists())

    def test_tampered_completed_file_is_rejected_without_replacing_it(self):
        output=self.root/'tampered';prepare(self.qa,self.cot,output,True)
        changed=output/'instruct.yaml';changed.write_text('modified by another process')
        with self.assertRaisesRegex(ValueError,'Conflicting/incomplete'):
            prepare(self.qa,self.cot,output,True)
        self.assertEqual(changed.read_text(),'modified by another process')


if __name__=='__main__':unittest.main()
