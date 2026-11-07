"""Regression for generated-program failure, reviewed migration and durable restart."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_cot
from construction_io import (BUILTINS, GeneratedProgramError, compile_program,
                             execute_program, latest_records, migrate_generated_failures,
                             records, recover_journal, reviewed_code_migration,
                             tool_sources_sha256)


class FailureTests(unittest.TestCase):
    def setUp(self):
        folder=Path(__file__).resolve().parents[1]/'runtime/tmp'
        folder.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='failure-regression-',dir=folder)
        self.root=Path(self.temp.name)

    def tearDown(self):self.temp.cleanup()

    def test_unbound_local_is_rejected_at_generated_code_boundary(self):
        compiled,_=compile_program('def execute_command(video_clip, query, possible_answers):\n'
                                   '    if query == "other":\n        actions = []\n'
                                   '    return actions')
        with self.assertRaisesRegex(GeneratedProgramError,'UnboundLocalError'):
            execute_program(compiled,{'__builtins__':BUILTINS},None,'actual',[])

    def test_tool_failures_and_memory_errors_are_not_generated_data_errors(self):
        compiled,_=compile_program('def execute_command(video_clip, query, possible_answers):\n'
                                   '    return Video_summary(video_clip, query)')
        for error in [RuntimeError('model backend failed'),NameError('broken tool implementation'),MemoryError('allocation failed')]:
            def broken(*_,error=error):raise error
            with self.assertRaises(type(error)) as caught:
                execute_program(compiled,{'__builtins__':BUILTINS,'Video_summary':broken},None,'q',[])
            self.assertIs(caught.exception,error)

    def test_reviewed_migration_rejects_unreviewed_code_or_changed_inputs(self):
        old=dict(driver_sha256='a'*64,io_sha256='b'*64,qa_sha256='q',program_prompt_sha256='p')
        new=dict(old,driver_sha256='c'*64,io_sha256='d'*64)
        policy=self.root/'compatibility.json'
        policy.write_text(json.dumps(dict(migrations=[dict(migration_id='reviewed-fix',reason='scoped generated failure',
            **{'from':{key:old[key] for key in ['driver_sha256','io_sha256']},
               'to':{key:new[key] for key in ['driver_sha256','io_sha256']}})])))
        self.assertEqual(reviewed_code_migration(old,new,policy)['migration_id'],'reviewed-fix')
        self.assertIsNone(reviewed_code_migration(old,dict(new,driver_sha256='x'*64),policy))
        self.assertIsNone(reviewed_code_migration(old,dict(new,qa_sha256='changed'),policy))
        self.assertIsNone(reviewed_code_migration(old,dict(new,program_prompt_sha256='changed'),policy))
        self.assertIsNone(reviewed_code_migration(old,dict(new,tool_sources_sha256={'tool.py':'changed'}),policy))

    def test_tool_fingerprint_tracks_code_and_prompts_and_requires_exact_review(self):
        agent=self.root/'agent';agent.mkdir()
        (agent/'tool.py').write_text('original implementation')
        (agent/'action.prompt').write_text('original prompt')
        original=tool_sources_sha256(agent)
        (agent/'tool.py').write_text('reviewed bounded input fix')
        current=tool_sources_sha256(agent)
        self.assertNotEqual(original,current)
        self.assertEqual(set(current),{'tool.py','action.prompt'})
        old=dict(driver_sha256='a',io_sha256='b',qa_sha256='same',tool_sources_sha256=original)
        new=dict(old,driver_sha256='c',io_sha256='d',tool_sources_sha256=current)
        policy=self.root/'policy.json'
        policy.write_text(json.dumps(dict(migrations=[dict(migration_id='tool-bounds',reason='reviewed tool fix',
            **{'from':{k:old[k] for k in ['driver_sha256','io_sha256','tool_sources_sha256']},
               'to':{k:new[k] for k in ['driver_sha256','io_sha256','tool_sources_sha256']}})])))
        self.assertEqual(reviewed_code_migration(old,new,policy)['migration_id'],'tool-bounds')
        (agent/'action.prompt').write_text('unreviewed prompt change')
        self.assertIsNone(reviewed_code_migration(old,dict(new,tool_sources_sha256=tool_sources_sha256(agent)),policy))

    def test_historical_reclassification_is_append_only_and_scoped(self):
        journal=self.root/'construct.jsonl'
        rows=[dict(sample_id='valid',status='ok',correct=True),
              dict(sample_id='generated',status='error',error='UnboundLocalError: actions',
                   traceback='  File "driver.py", line 1\n  File "<aotd-generated-program>", line 11\nUnboundLocalError: actions'),
              dict(sample_id='infrastructure',status='error',error='NameError: model',
                   traceback='  File "<aotd-generated-program>", line 1\n  File "model.py", line 2\nNameError: model')]
        journal.write_text(''.join(json.dumps(row)+'\n' for row in rows));original=journal.read_bytes()
        changed=migrate_generated_failures(journal,recover_journal(journal,metadata_only=True),dict(migration_id='reviewed-fix'))
        self.assertEqual(changed,1);self.assertTrue(journal.read_bytes().startswith(original))
        index=recover_journal(journal,metadata_only=True)
        self.assertEqual(index['generated']['status'],'rejected')
        self.assertEqual(index['infrastructure']['status'],'error')
        self.assertEqual(index['valid']['status'],'ok')
        self.assertEqual(migrate_generated_failures(journal,index,dict(migration_id='reviewed-fix')),0)

    def test_driver_restarts_after_infrastructure_failure_without_repeating_terminal_rows(self):
        for path in ['config/construction/program.prompt','config/construction/transfer.prompt',
                     'agent_system/prompts/cot_filter.prompt','tools/construction_io.py']:
            p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('isolated fixture')
        qa=self.root/'qa.jsonl';out=self.root/'output'
        qa.write_text(''.join(json.dumps(dict(sample_id=x))+'\n' for x in ['generated','infrastructure','tail']))
        calls=[];fail_once=[True]
        class Backend:
            def __init__(self,*_):pass
            def __call__(self,row,*_):
                sid=row['sample_id'];calls.append(sid)
                if sid=='generated':raise GeneratedProgramError('UnboundLocalError: fixture')
                if sid=='infrastructure' and fail_once[0]:
                    fail_once[0]=False
                    error=RuntimeError('recoverable backend failure')
                    error.program_code='exact failing program'
                    Path(_[0]).write_text('partial visual observations')
                    raise error
                return dict(correct=True)
        argv=['build_cot.py','--stage','construct','--qa',str(qa),'--output-dir',str(out)]
        with patch.object(build_cot,'ROOT',self.root),patch.object(build_cot,'AGENT',self.root/'agent_system'),\
             patch.object(build_cot,'ConstructionBackend',Backend),\
             patch.object(build_cot,'preflight',return_value=dict(records=3,ready=True)),\
             patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:build_cot.main()
            self.assertEqual(failure.exception.code,3)
            journal=out/'construct.jsonl';prefix=journal.read_bytes()
            failed=json.loads(prefix.splitlines()[-1])
            self.assertEqual(failed['code'],'exact failing program')
            self.assertEqual(failed['execution_trace'],'partial visual observations')
            build_cot.main();self.assertTrue(journal.read_bytes().startswith(prefix))
            self.assertEqual(calls,['generated','infrastructure','infrastructure','tail'])
            summary=json.loads((out/'construct.summary.json').read_text())
            self.assertEqual(summary['saved'],3);self.assertEqual(summary['statuses'],dict(rejected=1,ok=2))
            self.assertTrue(summary['complete_without_errors'])
            build_cot.main();self.assertEqual(len(calls),4)


if __name__=='__main__':unittest.main()
