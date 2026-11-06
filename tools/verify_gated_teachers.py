"""Target-node verification of non-secret download metadata and transferred weights."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'models/teachers/gated_manifest.json').read_text())
assert manifest['complete']
for row in manifest['files']:
    path=root/row['path']
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==row['sha256'] and path.stat().st_size==row['bytes'],path
print(f'[Llama transfer] SHA256 PASS: {len(manifest["files"])} files',flush=True)
