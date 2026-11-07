"""Simulate final export failure at the target step, then verify idempotent reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from keepalive_guard import gpu_pids,proc


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--training-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(os.environ['AOTD_ROOT']);output=args.training_dir.resolve();run=args.output_dir.resolve()
    run.mkdir(parents=True,exist_ok=True)
    python=str(root/'env/bin/python');keep=os.environ['AOTD_KEEPALIVE_SCRIPT']
    baseline={pid:proc(pid) for pid in gpu_pids()}
    if len(baseline)!=8:raise RuntimeError('Eight original GPU workers required')
    def hashes():
        result={}
        for p in sorted(output.glob('*.safetensors')):
            with p.open('rb') as stream:result[p.name]=hashlib.file_digest(stream,'sha256').hexdigest()
        return result
    before_hashes=hashes();assert before_hashes
    marker=output/'training_complete.json';assert marker.is_file()
    marker.replace(run/'training_complete.before.json')
    command=[python,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
             str(root/'distillation/train.py'),'--model_path',str(root/'models/LLaVA-7B-base'),
             '--data_yaml',str(root/'data/train_smoke/instruct.yaml'),'--output_dir',str(output),
             '--deepspeed',str(root/'distillation/configs/ds_config_zero2.json'),'--optim','adamw_torch',
             '--per_device_train_batch_size','1','--gradient_accumulation_steps','12','--max_steps','2',
             '--seed','42','--attn_implementation','sdpa','--dataloader_num_workers','0',
             '--save_strategy','steps','--save_steps','1','--save_total_limit','2','--resume_from_checkpoint','auto']
    def invoke(name):
        phase=run/name;phase.mkdir(exist_ok=True)
        wrapped=[python,'-B',str(root/'tools/keepalive_guard.py'),'--keepalive-script',keep,
                 '--state-dir',str(root/'runtime/guard-state'),'--',python,'-B',
                 str(root/'tools/with_spare_keepalive.py'),'--mode','all','--keepalive-script',keep,
                 '--state-dir',str(phase),'--','env','CUDA_VISIBLE_DEVICES=0,1,2,3',*command]
        with (run/(name+'.log')).open('w') as log:
            subprocess.run(wrapped,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if gpu_pids()==set(baseline) and all((p:=proc(pid)) and p['start']==old['start'] and p['state'] not in ('T','t','Z','X') for pid,old in baseline.items()):return
            time.sleep(.25)
        raise RuntimeError('Original eight workers did not restore exactly')
    print('[export recovery] checkpoint already at step 2; recover final publication',flush=True)
    invoke('recovered_export')
    metrics=json.loads((output/'training_metrics.json').read_text())
    assert metrics['global_step']==2 and metrics['new_optimizer_steps']==0,metrics
    assert metrics['final_export_recovered'] and marker.is_file()
    assert not (output/'checkpoint-3').exists()
    assert hashes()==before_hashes,'Final export changed learned weights'
    before_stats={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in output.glob('*.safetensors')}
    print('[completed skip] repeat the finished run without touching weights',flush=True)
    invoke('completed_skip')
    after_stats={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in output.glob('*.safetensors')}
    assert before_stats==after_stats
    report=dict(status='PASS',final_global_step=2,new_optimizer_steps=0,
                final_export_recovered=True,completion_marker_recreated=True,no_checkpoint_step_3=True,
                learned_weights_sha256_unchanged=True,completed_skip_weights_mtime_unchanged=True,
                original_workers=8,same_processes=True,duplicate_workers=0)
    (run/'completed_export_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
