"""Real distributed checkpoint interruption/resume and saved-model reload smoke.

Validation fixtures are isolated under this checkout and never exported as full data.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from keepalive_guard import gpu_pids, proc


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ['AOTD_ROOT'])
    run = args.output_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    python = str(root/'env/bin/python')
    keepalive = os.environ['AOTD_KEEPALIVE_SCRIPT']
    baseline = {pid:proc(pid) for pid in gpu_pids()}
    if len(baseline) != 8 or any(item is None or item['state'] in ('T','t','Z','X') for item in baseline.values()):
        raise RuntimeError('Eight active original workers are required before smoke')

    def restored():
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            if gpu_pids() == set(baseline) and all(
                (current:=proc(pid)) and current['start']==item['start'] and current['state'] not in ('T','t','Z','X')
                for pid,item in baseline.items()):
                return dict(original_workers=8,same_processes=True,all_active=True,duplicate_workers=0)
            time.sleep(.25)
        raise RuntimeError('The original eight workers were not restored exactly')

    subprocess.run([python,'-B',str(root/'tools/prepare_train_smoke.py')],check=True)
    output = run/'training'
    common = [python,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
              str(root/'distillation/train.py'),'--model_path',str(root/'models/LLaVA-7B-base'),
              '--data_yaml',str(root/'data/train_smoke/instruct.yaml'),'--output_dir',str(output),
              '--deepspeed',str(root/'distillation/configs/ds_config_zero2.json'),'--optim','adamw_torch',
              '--per_device_train_batch_size','1','--gradient_accumulation_steps','12','--max_steps','2',
              '--seed','42','--attn_implementation','sdpa','--dataloader_num_workers','0',
              '--save_strategy','steps','--save_steps','1','--save_total_limit','2']
    environment = dict(os.environ,AOTD_TRAIN_GPUS='0,1,2,3',AOTD_EVAL_GPU='0')

    def wrapped(command, name, mode):
        phase = run/name
        phase.mkdir(exist_ok=True)
        return [python,'-B',str(root/'tools/keepalive_guard.py'),'--keepalive-script',keepalive,
                '--state-dir',str(root/'runtime/guard-state'),'--',python,'-B',
                str(root/'tools/with_spare_keepalive.py'),'--mode',mode,'--keepalive-script',keepalive,
                '--state-dir',str(phase),'--','env',
                'CUDA_VISIBLE_DEVICES='+('0' if mode=='eval-only' else '0,1,2,3'),*command]

    print('[checkpoint smoke] save step 1, then interrupt the real distributed job',flush=True)
    marker = output/'checkpoint-1/aotd_checkpoint.json'
    with (run/'interrupted.log').open('w') as log:
        task = subprocess.Popen(wrapped([*common,'--resume_from_checkpoint','none'],'interrupted','all'),
                                env=environment,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+1200
            while not marker.is_file():
                if task.poll() is not None:
                    raise RuntimeError(f'Training exited before the complete step-1 checkpoint: {task.returncode}')
                if time.monotonic()>deadline:
                    raise TimeoutError('Timed out waiting for the atomic checkpoint completion marker')
                time.sleep(.1)
            checkpoint_before = json.loads(marker.read_text())
            task.send_signal(signal.SIGTERM)
            interrupted_exit = task.wait(timeout=90)
        finally:
            if task.poll() is None:
                task.terminate()
                try:task.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    task.kill();task.wait()
    if interrupted_exit != 143:
        raise RuntimeError(f'Expected signal interruption exit 143, got {interrupted_exit}')
    first_state = json.loads((output/'checkpoint-1/trainer_state.json').read_text())
    assert first_state['global_step']==1,first_state['global_step']
    first_restoration = restored()

    print('[checkpoint smoke] resume the same run from the complete step-1 checkpoint',flush=True)
    with (run/'resumed.log').open('w') as log:
        subprocess.run(wrapped([*common,'--resume_from_checkpoint','auto'],'resumed','all'),
                       env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    second_restoration = restored()
    final_state = json.loads((output/'trainer_state.json').read_text())
    metrics = json.loads((output/'training_metrics.json').read_text())
    assert final_state['global_step']==2,final_state['global_step']
    assert metrics['finite_loss'] and metrics['runtime_backend']['world_size']==4
    assert metrics['runtime_backend']['zero_stage']==2 and metrics['effective_batch_size']==48
    step_logs=[item for item in final_state['log_history'] if 'loss' in item]
    assert [item['step'] for item in step_logs]==[1,2],step_logs
    # A resumed optimizer/LR/RNG/data position must reproduce the known two-step fixture.
    for item,expected in zip(step_logs,[15.7559,15.8379]):
        assert abs(item['loss']-expected)<.1,(item,expected)
    complete = json.loads((output/'training_complete.json').read_text())
    second_marker = json.loads((output/'checkpoint-2/aotd_checkpoint.json').read_text())
    checkpoint_files=[str(path.relative_to(output/'checkpoint-1')) for path in (output/'checkpoint-1').rglob('*') if path.is_file()]
    assert sum('rng_state_' in path for path in checkpoint_files)==4,checkpoint_files
    assert sum('optim_states.pt' in path for path in checkpoint_files)==4,checkpoint_files

    print('[checkpoint smoke] reload the saved final model and processor for real evaluation',flush=True)
    eval_command=[python,'-u',str(root/'evaluation/eval_nextqa.py'),'--model_path',str(output),
                  '--data_dir',str(root/'data/nextqa/val.csv'),'--video_dir',str(root/'data/nextqa/NExTVideo'),
                  '--split','val','--limit','1','--attn_implementation','sdpa','--save_dir',str(run/'reload_eval.json')]
    with (run/'reload_eval.log').open('w') as log:
        subprocess.run(wrapped(eval_command,'reload','eval-only'),env=environment,stdout=log,
                       stderr=subprocess.STDOUT,check=True,timeout=600)
    third_restoration = restored()
    evaluation=json.loads((run/'reload_eval.json').read_text())
    assert evaluation['summary']['total']==1
    report=dict(status='PASS',snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='real checkpoint interruption/resume over isolated repeated QA; not full training',
                interruption=dict(exit_code=interrupted_exit,saved_global_step=first_state['global_step'],
                                  atomic_completion_marker=True,restoration=first_restoration),
                resume=dict(final_global_step=final_state['global_step'],logged_steps=[item['step'] for item in step_logs],
                            step_losses=[item['loss'] for item in step_logs],four_rank_rng_states=True,
                            four_rank_optimizer_states=True,restoration=second_restoration),
                optimizer_lr_rng_data_position=dict(known_fixture_loss_matches=True),
                final_model_reload=dict(total=evaluation['summary']['total'],invalid=evaluation['summary']['invalid'],
                                        restoration=third_restoration),
                world_size=4,effective_batch_size=48,zero_stage=2,
                final_completion_marker=bool(complete),checkpoint_file_count=len(checkpoint_files))
    (run/'checkpoint_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('[checkpoint smoke] PASS: interruption, optimizer continuation, final model reload and original worker restoration',flush=True)


if __name__=='__main__':main()
