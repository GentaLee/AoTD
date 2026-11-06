"""Data-loss, resume and execution-boundary regressions; fixtures never enter training."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from construction_io import (BUILTINS, answer_matches, compile_program, latest_records,
                             records, recover_journal)
from normalize_construction_sources import normalize


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'runtime/tmp'
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='construction-regression-',dir=base)
        self.directory=Path(self.temp.name)

    def tearDown(self):self.temp.cleanup()

    def test_large_record_crosses_read_boundaries(self):
        path=self.directory/'input.json'
        rows=[dict(sample_id='x',text='中"文'*40000),dict(sample_id='y',number=2)]
        path.write_text(json.dumps(rows,ensure_ascii=False))
        self.assertEqual(list(records(path)),rows)

    def test_torn_append_and_latest_attempt(self):
        path=self.directory/'journal.jsonl'
        path.write_bytes(b'{"sample_id":"a","status":"error"}\n{"sample_id":"a","status":"ok"}\n{"sam')
        index=recover_journal(path,metadata_only=True)
        self.assertEqual(index['a']['status'],'ok')
        self.assertEqual(list(latest_records(path,index)),[dict(sample_id='a',status='ok')])
        self.assertTrue(path.read_bytes().endswith(b'\n'))

    def test_mid_journal_corruption_is_not_silently_removed(self):
        path=self.directory/'bad.jsonl';path.write_text('broken\n{"sample_id":"a"}\n')
        with self.assertRaises(ValueError):recover_journal(path)

    def test_duplicate_official_nextqa_ids_preserve_both_questions(self):
        path=self.directory/'train.csv'
        fields=['video','qid','question','answer',*['a'+str(i) for i in range(5)]]
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader()
            for question in ['first real question','different real question']:
                writer.writerow(dict(video='1',qid='2',question=question,answer=0,**{'a'+str(i):str(i) for i in range(5)}))
        rows=list(normalize('NExT-QA',path,self.directory))
        self.assertEqual(len({x['sample_id'] for x in rows}),2)
        self.assertEqual({x['source_question_id'] for x in rows},{'1_2'})

    def test_star_text_answer_and_question_specific_clip(self):
        path=self.directory/'STAR_train.json'
        path.write_text(json.dumps([dict(question_id='Interaction_1',video_id='ABC',question='What?',answer='food',
                                        choices=[dict(choice_id=0,choice='water'),dict(choice_id=1,choice='food')],start=1.2,end=3.4)]))
        row=next(normalize('STAR',path,self.directory))
        self.assertEqual(row['answer'],1)
        self.assertTrue(row['video_path'].endswith('Interaction_1.mp4'))
        self.assertEqual(row['start_seconds'],1.2)

    def test_generated_code_cannot_import_or_access_process_internals(self):
        for body in ['import os\n    return os.environ','return video_clip.__class__','return open("/etc/passwd").read()',
                     'while True:\n        pass']:
            with self.assertRaises((ValueError,SyntaxError)):
                compile_program('def execute_command(video_clip, query, possible_answers):\n    '+body)
        compiled,_=compile_program('def execute_command(video_clip, query, possible_answers):\n    summary = Video_summary(video_clip, query)\n    return select_answer(query, {"summary": summary}, possible_answers)')
        namespace={'__builtins__':BUILTINS,'Video_summary':lambda *_:'recorded visual observation',
                   'select_answer':lambda *_:'A'}
        exec(compiled,namespace)
        self.assertEqual(namespace['execute_command'](None,'question',['a','b']),'A')

    def test_correct_answer_gate_has_no_random_or_substring_fallback(self):
        row=dict(answer=1,options=['apple','banana'])
        for text in ['B','(B)','banana']:self.assertTrue(answer_matches(text,row))
        for text in ['A','Answer might be B or A','unparseable']:self.assertFalse(answer_matches(text,row))

    def test_export_excludes_wrong_failed_and_rejected_records(self):
        qa=self.directory/'qa.jsonl';qa.write_text('')
        rows=[dict(sample_id=str(i),status=status,accepted=accepted,correct=correct,rationale='test fixture only')
              for i,(status,accepted,correct) in enumerate([('ok',True,True),('ok',False,True),('ok',True,False),('error',True,True)])]
        (self.directory/'filter.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('build_cot.py')),'--stage','export',
                        '--qa',str(qa),'--output-dir',str(self.directory)],check=True)
        self.assertEqual([x['sample_id'] for x in records(self.directory/'cot_filtered.jsonl')],['0'])

    def test_jsonl_join_preserves_qa_and_only_matching_cot(self):
        qa=self.directory/'qa.jsonl';cot=self.directory/'cot.jsonl';out=self.directory/'train'
        # This regression checks serialization/joining, not video decoding.
        media=self.directory/'fixture.mp4'
        media.write_bytes(b'isolated format fixture; never used for model training')
        rows=[dict(sample_id='fixture_mc',dataset='NExT-QA',video_id='fixture',video_path=str(media),
                   question='Fixture MC question',options=['red','blue'],answer=1),
              dict(sample_id='fixture_open',dataset='AGQA',video_id='fixture',video_path=str(media),
                   question='Fixture open question',options=[],answer='green')]
        qa.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        cot.write_text(json.dumps(dict(sample_id='fixture_mc',rationale='Fixture explanation only'))+'\n')
        subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('prepare_full_train.py')),
                        '--qa',str(qa),'--cot',str(cot),'--output-dir',str(out)],check=True)
        answering=list(records(out/'answering.json'));reasoning=list(records(out/'reasoning.json'))
        self.assertEqual([x['conversation'][1]['content'][0]['text'] for x in answering],['B','green'])
        self.assertEqual([x['sample_id'] for x in reasoning],['fixture_mc'])
        self.assertEqual(json.loads((out/'manifest.json').read_text())['qa_records'],2)

    def test_unknown_cot_id_never_publishes_training_files(self):
        qa=self.directory/'qa.jsonl';cot=self.directory/'cot.jsonl';out=self.directory/'train'
        qa.write_text('')
        cot.write_text(json.dumps(dict(sample_id='unknown_fixture',rationale='Fixture only'))+'\n')
        proc=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('prepare_full_train.py')),
                             '--qa',str(qa),'--cot',str(cot),'--output-dir',str(out)],capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertFalse((out/'answering.json').exists())
        self.assertFalse((out/'instruct.yaml').exists())


if __name__=='__main__':unittest.main()
