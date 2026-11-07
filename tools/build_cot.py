"""Offline Agent -> correct traces -> Llama transfer -> quality filter, with resume."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import struct
import sys
import threading
import time
import traceback
from construction_io import (BUILTINS, TOOLS, answer_matches, append_record, atomic_json,
                             compile_program, execute_program, GeneratedProgramError,
                             latest_records, records, recover_journal, sha256,
                             reviewed_code_migration, migrate_generated_failures, should_process)


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'agent_system'


class ProgressReporter:
    """Print bounded-rate status while the main thread loads models or runs a tool."""
    def __init__(self, stage, target, interval, statuses, saved):
        self.stage=stage;self.target=target;self.interval=interval
        self.started=time.monotonic();self.stop=threading.Event();self.lock=threading.Lock()
        self.state=dict(processed=0,new=0,saved=saved,ok=statuses['ok'],rejected=statuses['rejected'],
                        errors=statuses['error'],current='initializing')
        self.thread=threading.Thread(target=self._run,daemon=True)

    def update(self, **values):
        with self.lock:self.state.update(values)

    def _run(self):
        while not self.stop.wait(self.interval):
            with self.lock:state=dict(self.state)
            elapsed=time.monotonic()-self.started
            eta=f'{elapsed/state["new"]*max(self.target-state["processed"],0):.1f}s' if state['new'] else 'pending'
            print(f'[build {self.stage}] {state["processed"]}/{self.target} '
                  f'({100*state["processed"]/max(self.target,1):.1f}%) saved={state["saved"]} '
                  f'ok={state["ok"]} rejected={state["rejected"]} errors={state["errors"]} '
                  f'current={state["current"]} elapsed={elapsed:.1f}s estimated_remaining={eta}',flush=True)

    def __enter__(self):
        self.thread.start();return self

    def __exit__(self, *_):
        self.stop.set();self.thread.join()


def model_paths():
    base = ROOT / 'models/teachers'
    return dict(deepseek=base/'DeepSeek-Coder-6.7B-Instruct',
                llava=base/'LLaVA-Video-7B-DPO', owlv2=base/'OWLv2',
                llama=base/'Meta-Llama-3-8B-Instruct', transfer=base/'Llama-3.1-8B-Instruct',
                univtg=base/'UniVTG/model_best.ckpt', clip=base/'CLIP/ViT-B-32.pt')


def preflight(args):
    problems = []
    count = 0; missing_media = set(); seen = set()
    for row in records(args.qa):
        if row.get('split') != 'train':
            problems.append(f"Non-train sample: {row.get('sample_id')}")
        if row['sample_id'] in seen:
            raise ValueError('Duplicate normalized sample IDs')
        seen.add(row['sample_id']); count += 1
        if not Path(row['video_path']).is_file():
            missing_media.add(row['video_path'])
    if not count:
        problems.append('No QA records')
    if missing_media:
        problems.append(f'{len(missing_media)} train videos missing')
    paths = model_paths()
    needed = ['deepseek', 'llava', 'owlv2', 'llama', 'univtg', 'clip'] if args.stage in {'construct', 'check'} else ['transfer']
    if args.stage == 'check':
        needed.append('transfer')
    for name in needed:
        path = paths[name]
        if name in {'clip', 'univtg'}:
            if not path.is_file(): problems.append(f'Model missing: {path}')
        else:
            try:
                if not (path/'config.json').is_file():raise ValueError('config.json missing')
                index_path=path/'model.safetensors.index.json'
                shards=sorted(set(json.loads(index_path.read_text())['weight_map'].values())) if index_path.is_file() else ['model.safetensors']
                for shard in shards:
                    file=path/shard
                    with file.open('rb') as stream:
                        length=struct.unpack('<Q',stream.read(8))[0]
                        if not 0<length<64*1024**2:raise ValueError('invalid safetensors header length')
                        header=json.loads(stream.read(length))
                    expected=8+length+max(value['data_offsets'][1] for key,value in header.items() if key!='__metadata__')
                    if file.stat().st_size!=expected:raise ValueError(f'incomplete shard {shard}: {file.stat().st_size}/{expected} bytes')
            except (OSError,ValueError,KeyError,struct.error) as error:
                problems.append(f'Model missing/incomplete: {path}: {error}')
    for name in ['omegaconf', 'rich', 'ftfy', 'einops', 'timm', 'scipy', 'h5py', 'ffmpeg-python', 'tensorboardX', 'opencv-python-headless']:
        try: importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: problems.append(f'Agent package missing: {name}')
    report = dict(stage=args.stage, records=count, missing_media=sorted(missing_media), problems=problems,
                  ready=not problems, models={name:str(path) for name,path in paths.items()})
    atomic_json(args.output_dir/'preflight.json', report)
    print(json.dumps({key:value for key,value in report.items() if key not in {'models','missing_media'}}, indent=2), flush=True)
    return report


class ConstructionBackend:
    def __init__(self, output):
        import torch
        import numpy as np
        import random
        torch.manual_seed(20)
        torch.cuda.manual_seed_all(20)
        np.random.seed(20)
        random.seed(20)
        from omegaconf import OmegaConf
        paths = model_paths()
        config = OmegaConf.load(AGENT/'configs/base_config.yaml')
        for key in ['deepseek', 'llava', 'owlv2', 'llama']:
            config[key].model_path = str(paths[key])
        config.deepseek.prompt_path = str(ROOT/'config/construction/program.prompt')
        config.llava.action_prompt_path = str(AGENT/'prompts/action_single.prompt')
        config.select_answer_prompt_dir = str(AGENT/'prompts/select_answer.prompt')
        config.univtg.resume = str(paths['univtg'])
        config.univtg.clip_model_path = str(paths['clip'])
        config.univtg.save_dir = str(output/'univtg')
        config.cot.save_cot_dir = ''
        config_path = output/'agent_config.yaml'
        OmegaConf.save(config, config_path)
        os.environ['AOTD_AGENT_CONFIG'] = str(config_path)
        os.environ['AOTD_AGENT_STRICT'] = '1'
        sys.path.insert(0, str(AGENT))
        # UniVTG's legacy parser reads process argv; hide our own CLI arguments.
        previous_argv = sys.argv
        try:
            sys.argv = [previous_argv[0]]
            import video_clip
            import video_process
        finally:
            sys.argv = previous_argv
        self.clip_module = video_clip
        self.forward = video_process.forward
        self.torch = torch

    def __call__(self, row, trace_path, timeout):
        import decord
        import numpy as np
        reader = decord.VideoReader(row['video_path'], num_threads=1, ctx=decord.cpu(0))
        count = len(reader); fps = float(reader.get_avg_fps())
        if not count or fps <= 0: raise ValueError('Empty video or invalid fps')
        frames = reader.get_batch(np.linspace(0, count, 32, endpoint=False).astype(int)).asnumpy()
        clip_len = count / (32 * fps)
        video = self.clip_module.VideoClip(self.torch.from_numpy(frames), fps=1/clip_len, clip_len=clip_len)
        options = row.get('options') or None
        code = self.forward('deepseek', prompt=row['question'], possible_answers=options,
                            input_type='video_clip, query, possible_answers')
        # The old adapter added notebook display arguments; this driver does not use them.
        code = code.replace('video_clip, query, possible_answers, time_wait_between_lines, syntax',
                            'video_clip, query, possible_answers')
        compiled, code = compile_program(code)
        namespace = {'__builtins__': BUILTINS}
        namespace.update({name:getattr(self.clip_module, name) for name in TOOLS})
        self.clip_module.cot_name = str(trace_path)
        trace_path.write_text('')
        def expired(*_): raise TimeoutError(f'Generated program exceeded {timeout}s')
        previous = signal.signal(signal.SIGALRM, expired)
        try:
            signal.alarm(timeout)
            try:
                result = execute_program(compiled, namespace, video, row['question'], options)
            except GeneratedProgramError as error:
                error.program_code = code
                raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        if not isinstance(result, (str, int, float, bool)):
            raise ValueError('Program must return an answer, not a clip/container')
        trace = trace_path.read_text()
        visual_markers = ['call function Count', 'call function Video_query', 'call function Query_Actions',
                          'call function Query_Objs', 'filter action', 'call function Filter_frames_with_obj',
                          'call function Find', 'call function exist', 'call function Video_summary']
        if not any(marker in trace for marker in visual_markers):
            raise ValueError('Program did not record any visual tool observations')
        return dict(code=code, execution_trace=trace, program_output=str(result),
                    correct=answer_matches(result, row), sampled_frames=32)


class TextBackend:
    def __init__(self, stage):
        import torch
        from transformers import AutoTokenizer, pipeline
        path = model_paths()['transfer']
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer = tokenizer
        self.pipe = pipeline('text-generation', model=str(path), tokenizer=tokenizer,
                             device_map={'':'cuda:0'}, torch_dtype=torch.bfloat16)
        self.stage = stage
        self.prompt = (ROOT/'config/construction/transfer.prompt' if stage == 'transfer'
                       else AGENT/'prompts/cot_filter.prompt').read_text()

    def __call__(self, row, _trace_path, _timeout):
        prompt = self.prompt.replace('INSERT_QUESTION_HERE', row['question'])
        if self.stage == 'transfer':
            prompt = prompt.replace('INSERT_PROGRAM_HERE', row['code']).replace(
                'INSERT_EXECUTION_TRACE_HERE', row['execution_trace']+'\nprogram output: '+row['program_output'])
        else:
            prompt = prompt.replace('INSERT_RATIONALE_HERE', row['rationale'])
        text = self.tokenizer.apply_chat_template([
            {'role':'system','content':'You are a helpful assistant.'},
            {'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True)
        eos = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        result = self.pipe(text, return_full_text=False, do_sample=False,
                           max_new_tokens=2048 if self.stage == 'transfer' else 16,
                           eos_token_id=[x for x in eos if x is not None and x >= 0],
                           pad_token_id=self.tokenizer.eos_token_id)[0]['generated_text'].strip()
        if self.stage == 'transfer':
            import re
            rationale = re.sub(r'^rationale\s*:\s*', '', result, flags=re.I).strip()
            if not rationale: raise ValueError('Empty transferred rationale')
            return dict(rationale=rationale)
        # Accept only a true binary verdict, never arbitrary prose containing 'True'.
        return dict(filter_response=result, accepted=result.strip(' .\n\t').casefold() == 'true')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage', required=True, choices=['check', 'construct', 'transfer', 'filter', 'export'])
    parser.add_argument('--qa', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--limit', type=int, help='Explicit validation subset; never used by the full entrypoint')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A single writer per dataset; no duplicate GPU jobs or concurrently edited journals.
    with (args.output_dir/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_stage(args)


def run_stage(args):
    if args.stage == 'check':
        raise SystemExit(0 if preflight(args)['ready'] else 2)
    report = None if args.stage == 'export' else preflight(args)
    if report is not None and not report['ready']:raise SystemExit(2)
    if args.stage == 'export':
        path = args.output_dir/'filter.jsonl'
        accepted = recover_journal(path, metadata_only=True)
        count = 0
        temporary = args.output_dir/'cot_filtered.jsonl.partial'
        with temporary.open('w') as stream:
            for row in latest_records(path, accepted):
                if row.get('status') == 'ok' and row.get('accepted') and row.get('correct'):
                    stream.write(json.dumps(dict(sample_id=row['sample_id'], rationale=row['rationale']))+'\n')
                    count += 1
        temporary.replace(args.output_dir/'cot_filtered.jsonl')
        print(f'[export] {count} genuinely filtered CoT records', flush=True)
        return
    source = args.qa if args.stage == 'construct' else args.output_dir/('construct.jsonl' if args.stage == 'transfer' else 'transfer.jsonl')
    fingerprint = dict(qa_sha256=sha256(args.qa), stage=args.stage, input_sha256=sha256(source),input_bytes=source.stat().st_size,
                       program_prompt_sha256=sha256(ROOT/'config/construction/program.prompt'),
                       transfer_prompt_sha256=sha256(ROOT/'config/construction/transfer.prompt'),
                       filter_prompt_sha256=sha256(AGENT/'prompts/cot_filter.prompt'),
                       driver_sha256=sha256(Path(__file__)), io_sha256=sha256(ROOT/'tools/construction_io.py'),
                       model_manifests={name:sha256(ROOT/'models/teachers'/name) for name in
                           ['download_manifest.json','auxiliary_manifest.json','local_llama_manifest.json','gated_manifest.json']
                           if (ROOT/'models/teachers'/name).is_file()},
                       limit=args.limit, protocol='upstream tools; reconstructed zero-shot prompts; not original author CoT')
    manifest = args.output_dir/f'{args.stage}.manifest.json'
    migration = None
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        fixed=lambda item:{key:value for key,value in item.items() if key not in {'input_sha256','input_bytes','input_revisions','code_migrations'}}
        if fixed(previous)!=fixed(fingerprint):
            without_code=lambda item:{key:value for key,value in fixed(item).items() if key not in {'driver_sha256','io_sha256'}}
            if without_code(previous)==without_code(fingerprint):
                migration=reviewed_code_migration(previous,fingerprint,ROOT/'reproduction/metadata/construction_resume_compatibility.json')
            if migration is None:
                raise ValueError('QA/code/prompt changed: use a new output directory to preserve provenance')
            history=args.output_dir/'manifest_history';history.mkdir(exist_ok=True)
            old_digest=sha256(manifest)
            backup=history/f'{args.stage}-{old_digest}.json'
            if not backup.exists():atomic_json(backup,previous)
        code_migrations=previous.get('code_migrations',[])
        if migration:code_migrations=code_migrations+[migration]
        if code_migrations:fingerprint['code_migrations']=code_migrations
        revisions=previous.get('input_revisions',[])
        if previous['input_sha256']!=fingerprint['input_sha256']:
            size=previous.get('input_bytes')
            if size is None or size>fingerprint['input_bytes']:
                raise ValueError('Earlier stage input was replaced/truncated; refusing stale resume')
            digest=hashlib.sha256()
            with source.open('rb') as stream:
                left=size
                while left:
                    block=stream.read(min(left,1024**2))
                    if not block:raise ValueError('Short input prefix')
                    digest.update(block);left-=len(block)
            if digest.hexdigest()!=previous['input_sha256']:
                raise ValueError('Earlier stage input changed in place; refusing stale resume')
            revisions=revisions+[dict(bytes=size,sha256=previous['input_sha256'])]
        if revisions:fingerprint['input_revisions']=revisions
    atomic_json(manifest, fingerprint)
    journal = args.output_dir/f'{args.stage}.jsonl'
    done = recover_journal(journal, metadata_only=True)
    if migration:
        revised=migrate_generated_failures(journal,done,migration)
        if revised:
            done=recover_journal(journal,metadata_only=True)
        print(f'[resume] reviewed code migration applied; preserved={len(done)} corrected_classifications={revised}',flush=True)
    # Keep only the newest attempt per ID when a preceding stage retried failures.
    eligible = None if args.stage == 'construct' else recover_journal(source, metadata_only=True)
    rows = records(source) if eligible is None else latest_records(source, eligible)
    target = report['records'] if eligible is None else sum(row.get('status')=='ok' and bool(row.get('correct')) for row in eligible.values())
    if args.limit is not None:target=min(target,args.limit)
    total = 0
    backend = None
    trace_dir = args.output_dir/'traces'; trace_dir.mkdir(exist_ok=True)
    started = time.monotonic(); new = 0
    progress_interval=float(os.environ.get('AOTD_PROGRESS_INTERVAL_SECONDS','30'))
    if progress_interval<=0:raise ValueError('AOTD_PROGRESS_INTERVAL_SECONDS must be positive')
    statuses = Counter(x['status'] for x in done.values())
    with ProgressReporter(args.stage,target,progress_interval,statuses,len(done)) as progress, journal.open('a') as stream:
        for row in rows:
            if args.stage != 'construct' and (row.get('status') != 'ok' or not row.get('correct')): continue
            total += 1
            if args.limit is not None and total > args.limit: break
            sid = row['sample_id']
            if not should_process(done.get(sid),args.retry_errors):continue
            progress.update(processed=total-1,current=sid)
            if backend is None:
                backend = ConstructionBackend(args.output_dir) if args.stage == 'construct' else TextBackend(args.stage)
            item = {key:value for key,value in row.items() if key not in {'status','error','traceback','seconds'}}
            before = time.monotonic()
            trace = trace_dir/(hashlib.sha256(sid.encode()).hexdigest()+'.txt')
            try:
                item.update(backend(row, trace, args.timeout), status='ok')
            except (ValueError, SyntaxError, TypeError, AttributeError, KeyError, IndexError, TimeoutError) as error:
                # Invalid generated programs/rationales are rejected data, not
                # a reason to abort a whole source. Infrastructure/model failures
                # remain errors and stop the stage for investigation.
                item.update(status='rejected',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
                if isinstance(error,GeneratedProgramError):
                    item.update(code=getattr(error,'program_code',''),rejection_kind='generated_program')
            except Exception as error:
                item.update(status='error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            item['seconds'] = round(time.monotonic()-before, 3)
            append_record(stream, item)
            if sid in done: statuses[done[sid]['status']] -= 1
            done[sid] = {key:item.get(key) for key in ['sample_id','status','correct','accepted']}
            statuses[item['status']] += 1
            new += 1
            progress.update(processed=total,new=new,saved=len(done),ok=statuses['ok'],
                            rejected=statuses['rejected'],errors=statuses['error'],current=sid)
            if item['status']=='error':break
    statuses = Counter(x['status'] for x in done.values())
    summary = dict(stage=args.stage, saved=len(done), newly_processed=new, statuses=dict(statuses),
                   target_records=target,
                   correct=sum(bool(x.get('correct')) for x in done.values()),
                   accepted=sum(bool(x.get('accepted')) for x in done.values()),
                   validation_limit=args.limit, complete_without_errors=not statuses['error'])
    atomic_json(args.output_dir/f'{args.stage}.summary.json', summary)
    print(json.dumps(summary), flush=True)
    if statuses['error']: raise SystemExit(3)


if __name__ == '__main__':
    main()
