"""Select stable full-run directories without overwriting earlier experiment state."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def signature(root, mode):
    split = os.environ.get('AOTD_EVAL_SPLIT', 'val')
    return dict(mode=mode, split=split,
                protocol_sha256=digest(root/'config/construction/sources.json'),
                eval_csv_sha256=digest(root/'data/nextqa'/f'{split}.csv'),
                training_devices=os.environ.get('AOTD_TRAIN_GPUS', '0,1,2,3'),
                effective_batch_size=48,
                construction_dir=str(Path(os.environ.get('AOTD_CONSTRUCTION_DIR',str(root/'data/construction'))).resolve()),
                qa_override=os.environ.get('AOTD_QA_SOURCE'),
                cot_override=os.environ.get('AOTD_COT_SOURCE'),
                training_pool=str(Path(os.environ.get('AOTD_TRAIN_DATA', str(root/'data/five_source_train'))).resolve()),
                video_dir=str(Path(os.environ.get('AOTD_VIDEO_DIR',str(root/'data/nextqa/NExTVideo'))).resolve()))


def select(root, mode, resume=None, new=False, check=False):
    root = Path(root).resolve()
    results = root/'results';results.mkdir(parents=True, exist_ok=True)
    runtime = root/'runtime';runtime.mkdir(parents=True, exist_ok=True)
    pointer = runtime/f'latest_{mode}_run.json'
    expected = signature(root, mode)
    reuse = bool(resume) or (not new and not check and pointer.is_file())
    if resume and new:
        raise ValueError('--resume and --new-run are mutually exclusive')
    if resume:
        run = Path(resume).resolve()
    elif not new and not check and pointer.is_file():
        run = Path(json.loads(pointer.read_text())['run_dir']).resolve()
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
        run = results/f'{"check" if check else "full"}_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}'
    if run.parent != results.resolve():
        raise ValueError('Resume directory must be a direct child of this project results directory')
    if reuse and not run.is_dir():
        raise ValueError('Selected run directory is missing; use --new-run rather than silently losing checkpoint history')
    state = run/'run_state.json'
    if run.exists():
        if not state.is_file():
            raise ValueError('Unmanaged legacy run directory: preserve it and start a new managed run; construction journals remain resumable')
        previous = json.loads(state.read_text())
        if previous['signature'] != expected:
            raise ValueError('Full-run inputs/settings changed; use --new-run to preserve existing checkpoints')
    else:
        run.mkdir()
        atomic_json(state, dict(signature=expected, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='selected'))
    if not check:
        atomic_json(pointer, dict(run_dir=str(run)))
    return run


def record(run, status, exit_code):
    path = Path(run)/'run_state.json'
    state = json.loads(path.read_text())
    state.update(status=status, exit_code=exit_code, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(path, state)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    choose = sub.add_parser('select')
    choose.add_argument('--root', type=Path, required=True)
    choose.add_argument('--mode', choices=['all','eval-only'], required=True)
    choose.add_argument('--resume')
    choose.add_argument('--new-run', action='store_true')
    choose.add_argument('--check', action='store_true')
    update = sub.add_parser('record')
    update.add_argument('--run',type=Path,required=True)
    update.add_argument('--status',required=True)
    update.add_argument('--exit-code',type=int,required=True)
    args = parser.parse_args()
    if args.command == 'select':
        print(select(args.root,args.mode,args.resume,args.new_run,args.check))
    else:
        record(args.run,args.status,args.exit_code)


if __name__ == '__main__':
    main()
