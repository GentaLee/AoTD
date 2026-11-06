import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'models/download_manifest.json').read_text())
if len({x['repo'] for x in manifest}) != 2:
    raise RuntimeError('Both author and base model manifests are required')
for i, item in enumerate(manifest, 1):
    path = root / item['path']
    print(f"[assets {i}/{len(manifest)}] SHA256 {item['path']}", flush=True)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert path.stat().st_size == item['size'] and actual == item['sha256'], path
print('[assets] all model files verified', flush=True)
data_root = root / 'data/nextqa'
media = json.loads((data_root/'smoke_source_manifest.json').read_text())
for item in media['files'] + media['video_files']:
    path = data_root / item['path']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256'], path
print('[assets] CSV and all four real smoke videos verified', flush=True)
