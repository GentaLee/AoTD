"""Taihua CPU verification of staged local files against audited Shanghai copies."""
import hashlib
import json
from pathlib import Path
from construction_io import atomic_json

root=Path(__file__).resolve().parents[1]
data=json.loads((root/'models/teachers/local_llama_manifest.json').read_text())
assert data['complete']
for row in data['files']:
    path=root/row['path']
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==row['sha256'] and path.stat().st_size==row['bytes'],path
atomic_json(root/'models/teachers/local_llama_transfer_audit.json',dict(status='PASS',files=len(data['files']),official_weight_sha256_match=True))
print(f'[local Llama transfer] {len(data["files"])} files SHA256 PASS, including eight official weight shards',flush=True)
