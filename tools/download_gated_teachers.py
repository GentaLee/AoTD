"""Shanghai only: use a locally saved token, pin revisions, verify model shards."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
token=Path('/volume/yzhao04/persist/huggingface-auth/token').read_text().strip()
api=HfApi(token=token)
directory=ROOT/'models/teachers'
manifest_path=directory/'gated_manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(models={},files=[])
jobs=[]
for repo in ['meta-llama/Meta-Llama-3-8B-Instruct','meta-llama/Llama-3.1-8B-Instruct']:
    revision=manifest['models'].get(repo)
    try:
        info=api.model_info(repo,revision=revision,files_metadata=True)
        # Public metadata alone is insufficient: verify an authenticated file read.
        hf_hub_download(repo,'config.json',revision=info.sha,token=token,local_dir=directory/repo.split('/')[-1])
    except Exception as error:
        print(f'[gated access denied] {repo}: {type(error).__name__}, HTTP={getattr(getattr(error,"response",None),"status_code",None)}',flush=True)
        raise SystemExit(2)
    manifest['models'][repo]=info.sha
    for file in info.siblings:
        name=file.rfilename
        if '/' in name or name.startswith('.'):continue
        if not name.endswith(('.safetensors','.json','.model','.txt')):continue
        jobs.append((repo,info.sha,file))
    print(f'[gated revision] {repo} {info.sha}',flush=True)

def save():
    temporary=manifest_path.with_suffix('.partial')
    temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(manifest_path)

manifest['complete']=False;save()
def one(job):
    repo,rev,file=job
    path=Path(hf_hub_download(repo,file.rfilename,revision=rev,token=token,local_dir=directory/repo.split('/')[-1]))
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if file.lfs and digest!=file.lfs.sha256:raise ValueError(f'Model publisher hash mismatch: {path.name}')
    if file.size is not None and path.stat().st_size!=file.size:raise ValueError('Model size mismatch')
    return dict(repo=repo,revision=rev,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest)

completed=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(one,job) for job in jobs]):
        try:row=future.result()
        except Exception as error:
            print(f'[gated download failed] {type(error).__name__}',flush=True);raise SystemExit(3)
        completed.append(row);manifest['files']=completed;save()
        print(f'[gated verified {len(completed)}/{len(jobs)}] {row["path"]} bytes={row["bytes"]}',flush=True)
manifest['complete']=True;save()
print('[gated download] All required Llama files verified',flush=True)
