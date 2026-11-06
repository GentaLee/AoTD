import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

root = Path(__file__).resolve().parents[1]
targets = [
    ('Zhengrongzz/AoTD-7B', 'f0e74be6caf57cf84d5819919abefd80a826d1fb', 'AoTD-7B-author'),
    ('llava-hf/LLaVA-NeXT-Video-7B-hf', 'f417d9edeb69a9eaa88283780332d00ce3c66593', 'LLaVA-7B-base'),
]
jobs = []
for repo, revision, dest in targets:
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision, (info.sha, revision)
    for f in info.siblings:
        if f.rfilename.startswith('.') or f.rfilename.lower().endswith(('.md', '.jpg', '.png')):
            continue
        jobs.append((repo, revision, dest, f))

def download(job):
    repo, revision, dest, f = job
    print(f'[download] {repo}@{revision} {f.rfilename} ({f.size} bytes)', flush=True)
    path = Path(hf_hub_download(repo, f.rfilename, revision=revision,
                local_dir=root / 'models' / dest))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if f.lfs:
        assert actual == f.lfs.sha256, (path.name, 'LFS SHA256 mismatch')
    print(f'[verified] {path.name} sha256={actual}', flush=True)
    return dict(repo=repo, revision=revision, path=str(path.relative_to(root)),
                size=path.stat().st_size, sha256=actual)

manifest = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(download, job) for job in jobs]):
        manifest.append(future.result())
        (root / 'models' / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
print('[download] all model files verified', flush=True)
