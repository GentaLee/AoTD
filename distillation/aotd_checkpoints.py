"""Atomic, provenance-checked training checkpoints for the supported Trainer recipe.

Resume requires optimizer/scheduler/RNG state, not only inference weights.
The completion marker is published after all distributed ranks finish saving.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import os
import re
import shutil
import time

import yaml


CHECKPOINT_MARKER = 'aotd_checkpoint.json'
COMPLETE_MARKER = 'training_complete.json'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_padding_plan(records, batch, sample_ids=None):
    """Complete the last accumulation cycle with deterministic real-record repeats."""
    if records<=0 or batch<=0:raise ValueError('Padding requires a nonempty dataset and positive batch')
    count=(-records)%batch
    indices=[index%records for index in range(count)]
    if sample_ids is not None and len(sample_ids)!=count:raise ValueError('Padding ID count mismatch')
    return dict(method='repeat first rows of seeded shuffled training view; source files unchanged',
                original_records=records,padding_records=count,padded_records=records+count,
                effective_batch_size=batch,selected_indices=indices,
                selected_ids_sha256=hashlib.sha256(json.dumps(sample_ids or [],ensure_ascii=False).encode()).hexdigest())


def training_identity(cli, args, padding=None):
    """Bind resumed optimizer state to the same inputs, base and training recipe."""
    data = Path(cli.data_yaml).resolve()
    sources = []
    for item in yaml.safe_load(data.read_text())['datasets']:
        path = Path(item['json_path'])
        if not path.is_absolute():path = data.parent/path
        sources.append(dict(path=str(path.resolve()), sha256=digest(path)))
    model = Path(cli.model_path).resolve()
    model_files = {}
    for path in sorted(model.iterdir()):
        if not path.is_file():continue
        if path.suffix in {'.safetensors', '.bin'}:
            stat = path.stat()
            model_files[path.name] = dict(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        elif path.name in {'config.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json'}:
            model_files[path.name] = dict(sha256=digest(path))
    model_manifest = model.parent/'download_manifest.json'
    protocol = Path(__file__).resolve().parents[1]/'config/construction/sources.json'
    identity=dict(schema=1, model_path=str(model), model_files=model_files,
                model_manifest_sha256=digest(model_manifest) if model_manifest.is_file() else None,
                data_yaml_sha256=digest(data), sources=sources,
                protocol_sha256=digest(protocol) if protocol.is_file() else None,
                training=dict(world_size=args.world_size, microbatch=args.per_device_train_batch_size,
                              accumulation=args.gradient_accumulation_steps, max_steps=args.max_steps,
                              epochs=args.num_train_epochs, seed=args.seed, optimizer=args.optim.value,
                              learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio,
                              scheduler=args.lr_scheduler_type.value, attention=cli.attn_implementation,
                              deepspeed_sha256=digest(cli.deepspeed) if cli.deepspeed else None,
                              loss_normalization='trainer_per_microbatch_mean', bf16=True))
    if padding is not None:identity['training_view_padding']=padding
    return identity


def check_weights(folder):
    folder = Path(folder)
    if not (folder/'config.json').is_file():raise ValueError('Model config is missing')
    index = folder/'model.safetensors.index.json'
    binary_index = folder/'pytorch_model.bin.index.json'
    if index.is_file():files = set(json.loads(index.read_text())['weight_map'].values())
    elif binary_index.is_file():files = set(json.loads(binary_index.read_text())['weight_map'].values())
    else:files = [name for name in ['model.safetensors','pytorch_model.bin'] if (folder/name).is_file()]
    if not files:raise ValueError('Saved model weights are missing')
    for name in files:
        path=folder/name
        if not path.is_file() or path.stat().st_size<=0:raise ValueError(f'Model shard missing: {name}')
    for name in ['preprocessor_config.json','tokenizer_config.json']:
        if not (folder/name).is_file():raise ValueError(f'Processor file missing: {name}')


def checkpoint_components(folder, world_size, deepspeed):
    folder = Path(folder)
    state = json.loads((folder/'trainer_state.json').read_text())
    step = int(state['global_step'])
    if folder.name != f'checkpoint-{step}' or step<=0:raise ValueError('Checkpoint step mismatch')
    check_weights(folder)
    for rank in range(world_size):
        name = f'rng_state_{rank}.pth' if world_size>1 else 'rng_state.pth'
        path=folder/name
        if not path.is_file() or path.stat().st_size<=0:raise ValueError(f'RNG state missing: {name}')
    if deepspeed:
        tag = (folder/'latest').read_text().strip()
        if tag != f'global_step{step}':raise ValueError('DeepSpeed checkpoint step mismatch')
        directory=folder/tag
        # This integration is intentionally scoped to four-rank ZeRO-2 BF16.
        for rank in range(world_size):
            path=directory/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
            if not path.is_file() or path.stat().st_size<=0:raise ValueError(f'Optimizer shard missing: rank {rank}')
        model=directory/'mp_rank_00_model_states.pt'
        if not model.is_file() or model.stat().st_size<=0:raise ValueError('DeepSpeed model/scheduler state missing')
    else:
        for name in ['optimizer.pt','scheduler.pt']:
            path=folder/name
            if not path.is_file() or path.stat().st_size<=0:raise ValueError(f'Training state missing: {name}')
    return state


def publish_checkpoint(folder, identity, world_size, deepspeed, runtime_backend=None):
    folder = Path(folder)
    state = checkpoint_components(folder, world_size, deepspeed)
    files = {str(path.relative_to(folder)):path.stat().st_size for path in folder.rglob('*')
             if path.is_file() and path.name!=CHECKPOINT_MARKER and '.partial' not in path.name}
    marker=dict(schema=1,global_step=state['global_step'],world_size=world_size,
                deepspeed=bool(deepspeed),identity=identity,files=files,runtime_backend=runtime_backend)
    atomic_json(folder/CHECKPOINT_MARKER,marker)
    return marker


def validate_checkpoint(folder, identity):
    folder=Path(folder)
    marker=json.loads((folder/CHECKPOINT_MARKER).read_text())
    if marker.get('identity')!=identity:raise ValueError('Checkpoint inputs/base/training recipe changed')
    state=checkpoint_components(folder,marker['world_size'],marker['deepspeed'])
    if state['global_step']!=marker['global_step']:raise ValueError('Trainer step changed after checkpoint publication')
    for name,size in marker['files'].items():
        path=folder/name
        if not path.is_file() or path.stat().st_size!=size:raise ValueError(f'Incomplete checkpoint file: {name}')
    return marker


def choose_checkpoint(output, identity, requested='auto'):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    # Every distributed rank selects a checkpoint before constructing Trainer.
    # Serialize recovery so only one rank moves an incomplete save to history.
    with (output/'.checkpoint-selection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if requested=='auto':finish_pending_checkpoint_recovery(output,identity)
        return _choose_checkpoint(output,identity,requested)


def finish_checkpoint_recovery(output, history, recovery):
    """Finish directory moves even if the selector itself was interrupted."""
    for name,item in recovery['checkpoints'].items():
        if not re.fullmatch(r'checkpoint-\d+',name):raise ValueError('Invalid checkpoint recovery name')
        path=output/name;destination=Path(item['destination'])
        if destination.resolve()!=((history/name).resolve()):raise ValueError('Invalid checkpoint recovery destination')
        if path.exists():
            if destination.exists() or (path/CHECKPOINT_MARKER).exists() or (path/CHECKPOINT_MARKER).is_symlink():
                raise ValueError('Checkpoint changed during incomplete-save recovery; preserving files')
            path.replace(destination)
        elif not destination.is_dir():
            raise ValueError('Interrupted checkpoint recovery directory is missing; preserving history')
        item['moved']=True
        atomic_json(history/'recovery.json',recovery)
    recovery['status']='restart_from_zero'
    atomic_json(history/'recovery.json',recovery)
    atomic_json(output/'checkpoint_restart.json',recovery)


def finish_pending_checkpoint_recovery(output, identity):
    for marker in sorted((output/'checkpoint_history').glob('incomplete-*/recovery.json')):
        recovery=json.loads(marker.read_text())
        if recovery.get('status')!='quarantining':continue
        provenance=output/'training_provenance.json'
        if (recovery.get('identity')!=identity or not provenance.is_file()
                or json.loads(provenance.read_text())!=identity):
            raise ValueError('Interrupted checkpoint recovery provenance changed; preserving files')
        finish_checkpoint_recovery(output,marker.parent,recovery)


def _choose_checkpoint(output, identity, requested):
    if requested=='none':
        if (output/COMPLETE_MARKER).exists() or list(output.glob('checkpoint-*')):
            raise ValueError('Refusing to restart over existing training state; use a fresh output directory')
        return None
    if requested!='auto':
        path=Path(requested).resolve();validate_checkpoint(path,identity)
        return path
    candidates=sorted((path for path in output.glob('checkpoint-*') if re.fullmatch(r'checkpoint-\d+',path.name)),
                      key=lambda path:int(path.name.split('-')[-1]),reverse=True)
    problems=[]
    for path in candidates:
        try:validate_checkpoint(path,identity)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            problems.append(f'{path.name}: {exc}');continue
        if problems:print('[checkpoint] ignoring incomplete newer checkpoints: '+'; '.join(problems),flush=True)
        return path
    if candidates:
        provenance=output/'training_provenance.json'
        verified=(provenance.is_file() and json.loads(provenance.read_text())==identity)
        unpublished=all(path.is_dir() and not (path/CHECKPOINT_MARKER).exists()
                        and not (path/CHECKPOINT_MARKER).is_symlink() for path in candidates)
        if verified and unpublished:
            # No committed optimizer state exists, so there is nothing to
            # resume. Preserve the interrupted first save and explicitly
            # restart from step zero under the same verified provenance.
            history=output/'checkpoint_history'/f'incomplete-{time.time_ns()}-{os.getpid()}'
            history.mkdir(parents=True)
            recovery=dict(schema=1,reason='No complete checkpoint existed; interrupted unpublished saves preserved',
                          identity=identity,restart_from_global_step=0,resumed_from_checkpoint=None,
                          status='quarantining',checkpoints={path.name:dict(destination=str(history/path.name),
                              moved=False) for path in candidates},validation_errors=problems)
            atomic_json(history/'recovery.json',recovery)
            atomic_json(output/'checkpoint_restart.json',recovery)
            finish_checkpoint_recovery(output,history,recovery)
            print(f'[checkpoint] no committed checkpoint; restarting from step 0; interrupted saves preserved: {history}',flush=True)
            return None
        raise ValueError('No complete compatible checkpoint; preserving files: '+'; '.join(problems))
    return None


def validate_completion(output, identity):
    output=Path(output)
    marker=json.loads((output/COMPLETE_MARKER).read_text())
    if marker.get('identity')!=identity:raise ValueError('Completed training inputs/base/recipe changed')
    check_weights(output)
    metrics=json.loads((output/'training_metrics.json').read_text())
    if not metrics.get('finite_loss') or metrics['global_step']!=marker['global_step']:
        raise ValueError('Completed training metrics do not match marker')
    if marker['global_step']<marker['target_steps']:raise ValueError('Training stopped before its target')
    for name,size in marker['files'].items():
        path=output/name
        if not path.is_file() or path.stat().st_size!=size:raise ValueError(f'Incomplete final artifact: {name}')
    return marker


def publish_completion(output, identity, global_step, target_steps):
    output=Path(output)
    if global_step<target_steps:raise ValueError('Training stopped before completing the target')
    check_weights(output)
    names=['config.json','preprocessor_config.json','tokenizer_config.json','training_metrics.json']
    for pattern in ['*.safetensors','model.safetensors.index.json','pytorch_model*.bin','pytorch_model.bin.index.json']:
        names.extend(path.name for path in output.glob(pattern))
    marker=dict(schema=1,identity=identity,global_step=global_step,target_steps=target_steps,
                files={name:(output/name).stat().st_size for name in set(names)})
    atomic_json(output/COMPLETE_MARKER,marker)


def export_finished_checkpoint(output, checkpoint, identity):
    """Finish a failed final export without running another optimizer step."""
    output=Path(output);checkpoint=Path(checkpoint)
    marker=validate_checkpoint(checkpoint,identity)
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    target=int(state['max_steps']);step=int(state['global_step'])
    if target<=0 or step<target:raise ValueError('Checkpoint has unfinished optimizer steps')
    started=time.monotonic()
    losses=[float(row['loss']) for row in state['log_history'] if 'loss' in row]
    import math
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError('Finished checkpoint lacks finite saved loss history')
    output.mkdir(parents=True,exist_ok=True)
    for source in checkpoint.iterdir():
        if not source.is_file() or source.name==CHECKPOINT_MARKER:continue
        if source.suffix not in {'.json','.safetensors','.model','.txt','.tiktoken','.bin'}:continue
        if source.name=='training_args.bin':continue
        if source.name=='trainer_state.json':continue
        temporary=output/(source.name+'.partial')
        if temporary.exists():temporary.unlink()
        try:os.link(source,temporary)
        except OSError:shutil.copy2(source,temporary)
        temporary.replace(output/source.name)
    metrics=dict(global_step=step,target_steps=target,train_loss=sum(losses)/len(losses),
                 finite_loss=True,step_losses=losses,world_size=marker['world_size'],
                 runtime_backend=marker.get('runtime_backend'),
                 effective_batch_size=(identity['training']['world_size']*identity['training']['microbatch']*
                                       identity['training']['accumulation']),
                 loss_normalization='trainer_per_microbatch_mean',model_accepts_loss_kwargs=False,
                 resume_from_checkpoint=str(checkpoint),resumed_from_global_step=step,
                 new_optimizer_steps=0,final_export_recovered=True,
                 training_view_padding=identity.get('training_view_padding'),
                 final_export_seconds=time.monotonic()-started,
                 numeric_logs=[dict(step=row.get('step'),metrics={key:float(value) for key,value in row.items()
                    if key!='step' and isinstance(value,(int,float)) and not isinstance(value,bool)})
                    for row in state['log_history']])
    atomic_json(output/'training_metrics.json',metrics)
    atomic_json(output/'trainer_state.json',state)
    publish_completion(output,identity,step,target)
    return metrics


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--completed-output',type=Path,required=True)
    args=parser.parse_args()
    identity=json.loads((args.completed_output/'training_provenance.json').read_text())
    validate_completion(args.completed_output,identity)
    print('[checkpoint] completed training output verified',flush=True)


if __name__=='__main__':main()
