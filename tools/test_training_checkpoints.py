"""Exercise resumable-state integrity; fixtures contain no model/training tensors."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'distillation'))
from aotd_checkpoints import (CHECKPOINT_MARKER, atomic_json, choose_checkpoint,
                             export_finished_checkpoint,make_padding_plan,publish_checkpoint, publish_completion,
                             training_identity,validate_checkpoint,validate_completion)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'runtime/tmp'
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base,prefix='checkpoint-regression-')
        self.root=Path(self.temp.name)
        self.identity=dict(schema=1,model_files={'config.json':'fixture'},sources=[{'sha256':'fixed'}],
                           training=dict(world_size=4,max_steps=2,microbatch=1,accumulation=12))

    def tearDown(self):self.temp.cleanup()

    def weights(self,path):
        path.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','preprocessor_config.json','tokenizer_config.json']:
            (path/name).write_text('{}')
        (path/'model.safetensors').write_bytes(b'isolated integrity fixture, never loaded')

    def checkpoint(self,step,publish=True):
        path=self.root/f'checkpoint-{step}'
        self.weights(path)
        atomic_json(path/'trainer_state.json',dict(global_step=step,max_steps=2,
                    log_history=[dict(step=i,loss=1.0+i/10) for i in range(1,step+1)]))
        (path/'latest').write_text(f'global_step{step}')
        tag=path/f'global_step{step}';tag.mkdir()
        (tag/'mp_rank_00_model_states.pt').write_bytes(b'fixture model and scheduler state')
        for rank in range(4):
            (path/f'rng_state_{rank}.pth').write_bytes(b'fixture RNG')
            (tag/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'fixture optimizer state')
        if publish:publish_checkpoint(path,self.identity,4,True)
        return path

    def test_optimizer_rank_loss_rejects_checkpoint(self):
        path=self.checkpoint(1)
        (path/'global_step1/bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt').unlink()
        with self.assertRaisesRegex(ValueError,'Optimizer shard missing'):validate_checkpoint(path,self.identity)

    def test_newest_incomplete_save_falls_back_to_last_complete(self):
        older=self.checkpoint(1)
        newer=self.checkpoint(2,publish=False)
        self.assertEqual(choose_checkpoint(self.root,self.identity),older)
        self.assertTrue(newer.is_dir())
        self.assertFalse((newer/CHECKPOINT_MARKER).exists())

    def test_truncated_state_never_silently_resumes(self):
        path=self.checkpoint(1)
        (path/'rng_state_2.pth').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'Incomplete checkpoint file'):validate_checkpoint(path,self.identity)
        with self.assertRaisesRegex(ValueError,'No complete compatible'):choose_checkpoint(self.root,self.identity)

    def test_changed_data_or_world_size_rejects_optimizer_reuse(self):
        self.checkpoint(1)
        changed=copy.deepcopy(self.identity);changed['sources'][0]['sha256']='changed'
        with self.assertRaisesRegex(ValueError,'No complete compatible'):choose_checkpoint(self.root,changed)
        changed=copy.deepcopy(self.identity);changed['training']['world_size']=8
        with self.assertRaisesRegex(ValueError,'No complete compatible'):choose_checkpoint(self.root,changed)

    def test_inference_weights_are_not_a_resumable_training_checkpoint(self):
        path=self.root/'checkpoint-1';self.weights(path)
        with self.assertRaisesRegex(ValueError,'No complete compatible'):choose_checkpoint(self.root,self.identity)

    def test_completed_phase_needs_target_step_artifacts_and_same_identity(self):
        self.weights(self.root)
        atomic_json(self.root/'training_metrics.json',dict(global_step=2,finite_loss=True))
        with self.assertRaisesRegex(ValueError,'before completing'):publish_completion(self.root,self.identity,1,2)
        publish_completion(self.root,self.identity,2,2)
        validate_completion(self.root,self.identity)
        wrong=copy.deepcopy(self.identity);wrong['training']['max_steps']=3
        with self.assertRaisesRegex(ValueError,'recipe changed'):validate_completion(self.root,wrong)
        (self.root/'model.safetensors').write_bytes(b'truncated')
        with self.assertRaisesRegex(ValueError,'Incomplete final artifact'):validate_completion(self.root,self.identity)

    def test_explicit_fresh_start_refuses_existing_checkpoint(self):
        self.checkpoint(1)
        with self.assertRaisesRegex(ValueError,'fresh output'):choose_checkpoint(self.root,self.identity,'none')

    def test_finished_optimizer_state_recovers_final_export_without_new_steps(self):
        checkpoint=self.checkpoint(2)
        # A prior final export may have left a truncated output shard. The
        # complete training checkpoint remains authoritative and intact.
        (self.root/'model.safetensors').write_bytes(b'partial')
        metrics=export_finished_checkpoint(self.root,checkpoint,self.identity)
        self.assertEqual(metrics['new_optimizer_steps'],0)
        self.assertEqual(metrics['global_step'],2)
        self.assertEqual(metrics['step_losses'],[1.1,1.2])
        self.assertEqual((self.root/'model.safetensors').read_bytes(),(checkpoint/'model.safetensors').read_bytes())
        validate_completion(self.root,self.identity)
        validate_checkpoint(checkpoint,self.identity)

    def test_unfinished_checkpoint_cannot_publish_final_completion(self):
        checkpoint=self.checkpoint(1)
        with self.assertRaisesRegex(ValueError,'unfinished optimizer'):export_finished_checkpoint(self.root,checkpoint,self.identity)

    def test_full_pool_padding_consumes_every_source_row_and_keeps_complete_batches(self):
        count=133654;plan=make_padding_plan(count,48)
        consumed=list(range(count))+plan['selected_indices']
        self.assertEqual(plan['padding_records'],26)
        self.assertEqual(plan['padded_records'],133680)
        self.assertEqual(len(consumed)//48,2785)
        self.assertEqual(len(consumed)%48,0)
        self.assertEqual(set(consumed),set(range(count)))
        self.assertIn(count-1,consumed)
        self.assertEqual(make_padding_plan(96,48)['padding_records'],0)

    def test_padding_repeats_real_rows_even_for_a_tiny_dataset(self):
        plan=make_padding_plan(2,48)
        self.assertEqual(plan['selected_indices'],[0,1]*23)
        with self.assertRaises(ValueError):make_padding_plan(0,48)

    def test_padding_changes_provenance_only_when_explicitly_enabled(self):
        model=self.root/'base';model.mkdir();(model/'config.json').write_text('{}')
        source=self.root/'qa.json';source.write_text('[]')
        data=self.root/'input.yaml';data.write_text('datasets:\n  - json_path: qa.json\n')
        cli=SimpleNamespace(model_path=str(model),data_yaml=str(data),deepspeed=None,attn_implementation='sdpa')
        args=SimpleNamespace(world_size=4,per_device_train_batch_size=1,gradient_accumulation_steps=12,
                             max_steps=-1,num_train_epochs=1,seed=42,optim=SimpleNamespace(value='adamw_torch'),
                             learning_rate=4e-5,warmup_ratio=.03,lr_scheduler_type=SimpleNamespace(value='cosine'))
        original=training_identity(cli,args)
        self.assertNotIn('training_view_padding',original)
        padded=training_identity(cli,args,make_padding_plan(133654,48,[f'row-{i}' for i in range(26)]))
        self.assertNotEqual(original,padded)
        changed=training_identity(cli,args,make_padding_plan(133654,48,[f'changed-{i}' for i in range(26)]))
        self.assertNotEqual(padded,changed)


if __name__=='__main__':unittest.main()
