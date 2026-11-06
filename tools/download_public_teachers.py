"""network-enabled source host staging for public original Agent model dependencies."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

r=Path(__file__).resolve().parents[1]
targets=[
 ('deepseek-ai/deepseek-coder-6.7b-instruct','e5d64addd26a6a1db0f9b863abf6ee3141936807','DeepSeek-Coder-6.7B-Instruct'),
 ('google/owlv2-base-patch16-ensemble','cfd3195ba4ea9592eec887ded089f4c08eff231d','OWLv2'),
 ('llava-hf/LLaVA-NeXT-Video-7B-DPO-hf','f15ee7af0abdd85675ca947806dbefbb36a85d54','LLaVA-Video-7B-DPO')]
jobs=[]
for repo,rev,name in targets:
 info=HfApi().model_info(repo,revision=rev,files_metadata=True);assert info.sha==rev
 safe=any(f.rfilename.endswith('.safetensors') for f in info.siblings)
 for f in info.siblings:
  fn=f.rfilename
  if fn.startswith('.') or fn.endswith(('.md','.png','.jpg','.msgpack','.h5','.ot')):continue
  if fn.endswith('.bin') and (safe or not fn.startswith('pytorch_model')):continue
  if fn.endswith('.bin.index.json') and safe:continue
  if not fn.endswith(('.safetensors','.bin','.json','.model','.txt','.py')):continue
  jobs.append((repo,rev,name,f))
def download(job):
 repo,rev,name,f=job
 print(f'[teacher download] {repo} {f.rfilename}',flush=True)
 path=Path(hf_hub_download(repo,f.rfilename,revision=rev,local_dir=r/'models/teachers'/name))
 with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 if f.lfs:assert sha==f.lfs.sha256,path
 return dict(repo=repo,revision=rev,path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=sha)
records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for future in as_completed([pool.submit(download,j) for j in jobs]):
  row=future.result();records.append(row)
  (r/'models/teachers/download_manifest.json').write_text(json.dumps(dict(complete=len(records)==len(jobs),files=records),indent=2))
  print(f'[teacher verified {len(records)}/{len(jobs)}] {row["path"]}',flush=True)
print('[teachers] Public model download COMPLETE',flush=True)
