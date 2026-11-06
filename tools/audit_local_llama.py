"""network-enabled source host GPU Pod CPU audit of existing local models; read-only source files."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import argparse
import os

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--llama3-source',default=os.environ.get('AOTD_LLAMA3_SOURCE'),type=Path)
parser.add_argument('--llama31-source',default=os.environ.get('AOTD_LLAMA31_SOURCE'),type=Path)
args=parser.parse_args()
if not args.llama3_source or not args.llama31_source:parser.error('Both authorized local model paths are required')
TARGETS=[('Meta-Llama-3-8B-Instruct','meta-llama/Meta-Llama-3-8B-Instruct',args.llama3_source),
         ('Llama-3.1-8B-Instruct','meta-llama/Llama-3.1-8B-Instruct',args.llama31_source)]
publisher_path=ROOT/'downloads/llama_publisher_hashes.json'
if not publisher_path.is_file():publisher_path=ROOT/'reproduction/metadata/llama_publisher_hashes.json'
publisher=json.loads(publisher_path.read_text())
manifest=dict(complete=False,models=[],files=[])
for folder,repo,source in TARGETS:
    config=json.loads((source/'config.json').read_text())
    index=json.loads((source/'model.safetensors.index.json').read_text())
    assert config['model_type']=='llama' and config['hidden_size']==4096 and config['num_hidden_layers']==32
    names=sorted(set(index['weight_map'].values()))
    def one(name):
        path=source/name
        with path.open('rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0]
            assert 0<length<64*1024**2
            header=json.loads(stream.read(length))
        dtypes=Counter();keys=set()
        for key,tensor in header.items():
            if key=='__metadata__':continue
            assert index['weight_map'][key]==name
            start,end=tensor['data_offsets'];assert 0<=start<=end<=path.stat().st_size-8-length
            count=1
            for dimension in tensor['shape']:count*=dimension
            dtypes[tensor['dtype']]+=count;keys.add(key)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        expected=publisher.get(repo,{}).get('files',{}).get(name,{}).get('sha256')
        assert expected and expected==digest,f'Official publisher SHA256 differs: {repo}/{name}'
        print(f'[local Llama audit] official SHA256 PASS {folder}/{name}',flush=True)
        return dict(path=f'models/teachers/{folder}/{name}',source=str(path),bytes=path.stat().st_size,
                    sha256=digest,publisher_sha256_match=True,dtype_parameters=dict(dtypes)),keys
    with ThreadPoolExecutor(max_workers=2) as pool:audited=list(pool.map(one,names))
    keys=set().union(*(keys for _,keys in audited));assert keys==set(index['weight_map'])
    dtype=Counter()
    for row,_ in audited:
        dtype.update(row['dtype_parameters']);manifest['files'].append(row)
    assert dtype['BF16']/sum(dtype.values())>0.99
    metadata=[]
    for path in source.iterdir():
        if path.is_file() and path.suffix!='.safetensors' and not path.name.startswith('.'):
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            row=dict(path=f'models/teachers/{folder}/{path.name}',source=str(path),bytes=path.stat().st_size,sha256=digest)
            manifest['files'].append(row);metadata.append(path.name)
    manifest['models'].append(dict(folder=folder,official_repo=repo,source=str(source),
        publisher_revision=publisher[repo]['revision'],official_weight_sha256_match=True,
        dtype_parameters=dict(dtype),metadata_files=metadata))
manifest['complete']=True
(ROOT/'models/teachers/local_llama_manifest.json').write_text(json.dumps(manifest,indent=2))
print('[local Llama audit] Both local models match official publisher weight hashes',flush=True)
