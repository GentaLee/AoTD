"""Regression checks for stable run selection and experiment isolation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from full_run_state import select, record


class FullRunTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'runtime/tmp'
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base,prefix='run-state-')
        self.root=Path(self.temp.name)
        (self.root/'config/construction').mkdir(parents=True)
        (self.root/'config/construction/sources.json').write_text('{"protocol_id":"fixture"}')
        (self.root/'data/nextqa').mkdir(parents=True)
        (self.root/'data/nextqa/val.csv').write_text('fixture-only')
        self.env=patch.dict(os.environ,{},clear=True);self.env.start()

    def tearDown(self):
        self.env.stop();self.temp.cleanup()

    def test_failed_run_reuses_checkpoint_directory_and_records_state(self):
        first=select(self.root,'all')
        record(first,'failed',23)
        (first/'instruct_s42/checkpoint-100').mkdir(parents=True)
        second=select(self.root,'all')
        self.assertEqual(first,second)
        self.assertTrue((second/'instruct_s42/checkpoint-100').is_dir())
        self.assertEqual(json.loads((second/'run_state.json').read_text())['exit_code'],23)

    def test_input_change_refuses_implicit_resume(self):
        select(self.root,'all')
        (self.root/'config/construction/sources.json').write_text('{"protocol_id":"changed"}')
        with self.assertRaisesRegex(ValueError,'changed'):select(self.root,'all')

    def test_check_does_not_replace_active_run_pointer(self):
        first=select(self.root,'all')
        check=select(self.root,'all',check=True)
        self.assertNotEqual(first,check)
        self.assertEqual(first,select(self.root,'all'))

    def test_external_and_unmanaged_directories_are_rejected(self):
        with self.assertRaises(ValueError):select(self.root,'all',resume=str(self.root))
        legacy=self.root/'results/legacy';legacy.mkdir()
        with self.assertRaisesRegex(ValueError,'legacy'):select(self.root,'all',resume=str(legacy))

    def test_modes_and_explicit_new_run_are_isolated(self):
        first=select(self.root,'all')
        evaluation=select(self.root,'eval-only')
        self.assertNotEqual(first,evaluation)
        # Filesystem timestamp granularity must not cause an existing run to be reused.
        (first/'keep.txt').write_text('preserve')
        new=select(self.root,'all',new=True)
        self.assertNotEqual(first,new)
        self.assertTrue((first/'keep.txt').is_file())

    def test_deleted_pointer_target_does_not_silently_restart(self):
        first=select(self.root,'all')
        (first/'run_state.json').unlink();first.rmdir()
        with self.assertRaisesRegex(ValueError,'missing'):select(self.root,'all')


if __name__=='__main__':unittest.main()
