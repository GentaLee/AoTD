"""Durable evaluation prefixes with immutable input and model fingerprints."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def fingerprint(model, csv, video_dir, script, split, limit, attention):
    model=Path(model).resolve()
    index=model/'model.safetensors.index.json'
    weights=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.is_file() else ['model.safetensors']
    # Hash weights rather than relying on path/mtime so replacing a model cannot
    # silently reuse answers produced by a different checkpoint.
    inference_files=['config.json','generation_config.json','model.safetensors.index.json',
                     'processor_config.json','preprocessor_config.json','tokenizer.json','tokenizer.model',
                     'tokenizer_config.json','special_tokens_map.json','added_tokens.json','chat_template.json',
                     'vocab.json','merges.txt']
    names=weights+[name for name in inference_files if (model/name).is_file()]
    versions={}
    for name in ['torch','transformers','accelerate','decord','numpy']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return dict(version=1,model_path=str(model),model_files={name:digest(model/name) for name in names},
                csv_sha256=digest(csv),video_dir=str(Path(video_dir).resolve()),script_sha256=digest(script),
                split=split,limit=limit,attention=attention,frames=32,max_new_tokens=1024,
                helper_sha256=digest(__file__),packages=versions,do_sample=False,seed=42,parsing='strict_letter_v1')


def summarize(rows):
    total=len(rows)
    result=dict(total=total,correct=sum(row['correct'] for row in rows),
                accuracy=sum(row['correct'] for row in rows)/total,
                invalid=sum(row['invalid'] for row in rows),
                invalid_rate=sum(row['invalid'] for row in rows)/total)
    for group in 'CTD':
        subset=[row for row in rows if row['category'].startswith(group)]
        result[group]=dict(total=len(subset),correct=sum(row['correct'] for row in subset),
                           accuracy=sum(row['correct'] for row in subset)/len(subset) if subset else None)
    return result


def recover_prefix(journal, selected, verify):
    journal=Path(journal)
    if not journal.exists():return []
    records=[]
    with journal.open('r+b') as stream:
        while True:
            offset=stream.tell();line=stream.readline()
            if not line:break
            try:record=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if line.endswith(b'\n') or stream.read(1):
                    raise ValueError('Corrupt committed evaluation record; only a torn final line can be recovered')
                stream.truncate(offset);stream.flush();os.fsync(stream.fileno());break
            if not line.endswith(b'\n'):
                stream.truncate(offset);stream.flush();os.fsync(stream.fileno());break
            index=len(records)
            if index>=len(selected):raise ValueError('Evaluation journal exceeds the selected dataset')
            verify(record,selected[index])
            records.append(record)
    return records


def prepare(output, expected, selected, verify):
    output=Path(output)
    journal=output.with_suffix('.jsonl')
    manifest=output.with_suffix('.resume.json')
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous.get('fingerprint')!=expected:
            raise ValueError('Evaluation inputs/model/settings changed; use a new output path')
    elif output.exists() or journal.exists():
        raise ValueError('Existing evaluation output has no verified resume manifest')
    else:atomic_json(manifest,dict(fingerprint=expected))
    rows=recover_prefix(journal,selected,verify)
    if output.exists():
        complete=json.loads(output.read_text())
        if len(rows)!=len(selected) or complete.get('res_list')!=rows:
            raise ValueError('Completed evaluation output does not match its durable journal')
        if complete.get('summary')!=summarize(rows):
            raise ValueError('Completed evaluation summary is inconsistent')
        return rows,complete
    return rows,None
