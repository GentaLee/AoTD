"""Data-loss, resume and execution-boundary regressions; fixtures never enter training."""
import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from construction_io import (BUILTINS, answer_matches, compile_program, latest_records,
                             records, recover_journal)
from normalize_construction_sources import normalize
from training_protocol import load_protocol, coverage_errors
from prepare_egoqa import convert as convert_egoqa
from build_cot import ProgressReporter


class ConstructionTests(unittest.TestCase):
    def test_progress_heartbeat_continues_during_blocked_sample(self):
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):
            with ProgressReporter('fixture',1,0.02,Counter(),0) as progress:
                progress.update(current='slow-fixture')
                threading.Event().wait(0.08)
        output=captured.getvalue()
        self.assertGreaterEqual(output.count('[build fixture]'),1)
        self.assertIn('current=slow-fixture',output)
        self.assertIn('estimated_remaining=pending',output)
        self.assertFalse(progress.thread.is_alive())

    def test_egoqa_preserves_native_nested_paths_multiple_questions_and_answers(self):
        media=self.directory/'videos'
        for folder in ['uuid-one','uuid-two']:
            (media/folder).mkdir(parents=True)
            (media/folder/'split_0.mp4').write_bytes(b'media existence fixture only')
        native=self.directory/'train.json';output=self.directory/'normalized.jsonl'
        answers=['Exact natural language answer.','  Keep original answer whitespace.  ','另一条原始答案']
        native.write_text(json.dumps([
            dict(video='uuid-one/split_0.mp4',QA=[dict(q='Question one?',a=answers[0]),dict(q='Question two?',a=answers[1])]),
            dict(video='uuid-two/split_0.mp4',QA=[dict(q='Question three?',a=answers[2])])],ensure_ascii=False))
        convert_egoqa(native,media,output)
        rows=list(records(output))
        self.assertEqual([row['answer'] for row in rows],answers)
        self.assertEqual([row['sample_id'] for row in rows],['EgoQA:uuid-one/split_0:0','EgoQA:uuid-one/split_0:1','EgoQA:uuid-two/split_0:0'])
        self.assertEqual(len({row['video_id'] for row in rows}),2)
        self.assertTrue(all(row['options']==[] and row['split']=='train' for row in rows))
        self.assertTrue(rows[0]['video_path'].endswith('uuid-one/split_0.mp4'))

    def test_egoqa_missing_media_never_publishes_partial_import(self):
        media=self.directory/'videos';media.mkdir()
        (media/'present.mp4').write_bytes(b'media existence fixture only')
        native=self.directory/'train.json';output=self.directory/'normalized.jsonl'
        native.write_text(json.dumps([dict(video=name,QA=[dict(q='Fixture?',a='Original')]) for name in ['present.mp4','missing.mp4']]))
        with self.assertRaises(FileNotFoundError):convert_egoqa(native,media,output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.source_audit.json').exists())
        output.write_text('previous verified import')
        with self.assertRaises(FileNotFoundError):convert_egoqa(native,media,output)
        self.assertEqual(output.read_text(),'previous verified import')

    def test_egoqa_audited_exclusions_preserve_all_retained_qa_and_source_ids(self):
        media=self.directory/'videos';media.mkdir();(media/'valid.mp4').write_bytes(b'file presence fixture only')
        native=self.directory/'train.json';output=self.directory/'normalized.jsonl';exclusions=self.directory/'exclusions.json'
        native.write_text(json.dumps([
            dict(video='audited-missing.mp4',QA=[dict(q='Excluded?',a='Recorded original')]),
            dict(video='valid.mp4',QA=[dict(q='First?',a=' Original first '),dict(q='Second?',a='Original second')])]))
        exclusions.write_text(json.dumps(dict(annotations_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            annotated_records=3,valid_videos=1,excluded_videos=1,original_archive_sha256='a'*64,
            exclusions=[dict(video='audited-missing.mp4',status='missing',reason='Explicit isolated audited fixture')])) )
        convert_egoqa(native,media,output,exclusions)
        rows=list(records(output))
        self.assertEqual([row['sample_id'] for row in rows],['EgoQA:valid:0','EgoQA:valid:1'])
        self.assertEqual([row['answer'] for row in rows],[' Original first ','Original second'])
        self.assertTrue(all(row['source_annotation_row']==1 for row in rows))
        audit=json.loads(output.with_suffix('.source_audit.json').read_text())
        self.assertEqual((audit['annotated_records'],audit['excluded_qa_records'],audit['records']),(3,1,2))
        self.assertEqual(audit['exclusions_manifest_sha256'],hashlib.sha256(exclusions.read_bytes()).hexdigest())

    def test_egoqa_audited_exclusions_reject_source_hash_mismatch_and_unknown_entries(self):
        media=self.directory/'videos';media.mkdir();(media/'valid.mp4').write_bytes(b'file presence fixture only')
        native=self.directory/'train.json';output=self.directory/'normalized.jsonl';exclusions=self.directory/'exclusions.json'
        native.write_text(json.dumps([dict(video='valid.mp4',QA=[dict(q='Fixture?',a='Original')])]))
        audit=dict(annotations_sha256='0'*64,annotated_records=1,valid_videos=1,
                   original_archive_sha256='a'*64,exclusions=[])
        exclusions.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'annotation SHA256'):convert_egoqa(native,media,output,exclusions)
        self.assertFalse(output.exists())
        audit.update(annotations_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                     exclusions=[dict(video='unknown.mp4',status='missing',reason='Not in source')])
        exclusions.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'absent from source'):convert_egoqa(native,media,output,exclusions)
        self.assertFalse(output.exists())

    def test_egoqa_exclusions_never_hide_unlisted_missing_video(self):
        media=self.directory/'videos';media.mkdir();(media/'valid.mp4').write_bytes(b'file presence fixture only')
        native=self.directory/'train.json';output=self.directory/'normalized.jsonl';exclusions=self.directory/'exclusions.json'
        native.write_text(json.dumps([dict(video=name,QA=[dict(q='Fixture?',a='Original')])
                                      for name in ['valid.mp4','audited.mp4','unexpected.mp4']]))
        exclusions.write_text(json.dumps(dict(annotations_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            annotated_records=3,valid_videos=2,original_archive_sha256='a'*64,
            exclusions=[dict(video='audited.mp4',status='missing',reason='Explicit isolated audited fixture')])) )
        with self.assertRaises(FileNotFoundError):convert_egoqa(native,media,output,exclusions)
        self.assertFalse(output.exists())

    def test_selected_protocol_excludes_anetqa_and_requires_each_source(self):
        protocol=load_protocol()
        counts={source['dataset']:source['minimum_qa_records'] for source in protocol['sources']}
        self.assertEqual(set(counts),{'NExT-QA','STAR','AGQA','CLEVRER','EgoQA'})
        self.assertEqual(coverage_errors(counts,protocol),[])
        self.assertTrue(coverage_errors(dict(counts,ANetQA=25000),protocol))
        self.assertTrue(coverage_errors({name:1 for name in counts},protocol))
        counts.pop('EgoQA')
        self.assertTrue(any('EgoQA' in message for message in coverage_errors(counts,protocol)))

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

    def test_full_preflight_checks_source_coverage_qa_identity_and_cot_join(self):
        root=self.directory/'preflight';root.mkdir()
        config=root/'config/construction';config.mkdir(parents=True)
        protocol=load_protocol()
        # Isolated small manifest tests policy logic, never model training.
        for source in protocol['sources']:source['minimum_qa_records']=1
        (config/'sources.json').write_text(json.dumps(protocol))
        media=root/'media';media.mkdir();(media/'fixture.mp4').write_bytes(b'preflight file-presence fixture only')
        annotations=root/'data/nextqa';annotations.mkdir(parents=True)
        (annotations/'val.csv').write_text('video\n'+'fixture\n'*4996)
        train=root/'data/five_source_train';train.mkdir()
        qa=[dict(sample_id=source['dataset']+':fixture',dataset=source['dataset'],
                 video_path=str(media/'fixture.mp4'),conversation=[dict(role='user')]) for source in protocol['sources']]
        cot=[dict(row) for row,source in zip(qa,protocol['sources']) if source['cot']]
        (train/'answering.json').write_text(json.dumps(qa))
        (train/'reasoning.json').write_text(json.dumps(cot))
        (train/'instruct.yaml').write_text('datasets:\n  - json_path: answering.json\n')
        (train/'aotd.yaml').write_text('datasets:\n  - json_path: answering.json\n  - json_path: reasoning.json\n')
        raw_qa=root/'qa-source.json';raw_cot=root/'cot-source.json'
        raw_qa.write_text('isolated source fingerprint fixture');raw_cot.write_text('isolated CoT source fingerprint fixture')
        def publish_manifest():
            digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            counts=Counter(row['dataset'] for row in qa)
            manifest=dict(protocol_id=protocol['protocol_id'],full_pool=True,qa_records=len(qa),cot_records=len(cot),
                          dataset_counts=dict(counts),input_sha256={str(path):digest(path) for path in [raw_qa,raw_cot]},
                          training_file_sha256={path.name:digest(path) for path in [train/'answering.json',train/'reasoning.json']})
            (train/'manifest.json').write_text(json.dumps(manifest))
        publish_manifest()
        env=dict(os.environ,AOTD_ROOT=str(root),AOTD_RUN_DIR=str(root/'results'),AOTD_VIDEO_DIR=str(media),
                 AOTD_TRAIN_DATA=str(train),AOTD_EVAL_SPLIT='val',AOTD_FULL_MODE='all')
        command=[sys.executable,'-B',str(Path(__file__).with_name('full_preflight.py'))]
        good=subprocess.run(command,env=env,capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stdout+good.stderr)
        raw_qa.write_text('updated source after training pool publication')
        stale=subprocess.run(command,env=env,capture_output=True,text=True)
        self.assertNotEqual(stale.returncode,0)
        self.assertIn('Training source changed after preparation',stale.stdout)
        self.assertIn('fresh AOTD_TRAIN_DATA',stale.stdout)
        publish_manifest()
        qa.pop();(train/'answering.json').write_text(json.dumps(qa))
        publish_manifest()
        missing=subprocess.run(command,env=env,capture_output=True,text=True)
        self.assertNotEqual(missing.returncode,0)
        self.assertIn('EgoQA',missing.stdout)
        qa.append(dict(qa[0],sample_id='excluded',dataset='ANetQA'))
        (train/'answering.json').write_text(json.dumps(qa))
        publish_manifest()
        excluded=subprocess.run(command,env=env,capture_output=True,text=True)
        self.assertNotEqual(excluded.returncode,0)
        self.assertIn('Unexpected/excluded',excluded.stdout)
        qa.pop();qa.append(dict(qa[0],dataset='EgoQA',sample_id='EgoQA:fixture'))
        (train/'answering.json').write_text(json.dumps(qa))
        cot[0]['sample_id']='unknown'
        (train/'reasoning.json').write_text(json.dumps(cot))
        publish_manifest()
        unmatched=subprocess.run(command,env=env,capture_output=True,text=True)
        self.assertNotEqual(unmatched.returncode,0)
        self.assertIn('no matching source QA',unmatched.stdout)


if __name__=='__main__':unittest.main()
