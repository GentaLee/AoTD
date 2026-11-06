"""Link only the image's GPU torch stack into the isolated venv."""
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys

dest = Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)
links = {}
for name in ['torch', 'torchvision', 'triton', 'nvidia', 'functorch', 'torchgen']:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(f'missing platform package: {name}')
    source = Path(next(iter(spec.submodule_search_locations)))
    links[name] = str(source)
for name in ['torch', 'torchvision', 'pytorch-triton']:
    source = Path(metadata.distribution(name)._path)
    links[source.name] = str(source)
for name, source in links.items():
    target = dest / name
    if target.is_symlink():
        if target.resolve() != Path(source).resolve():
            raise RuntimeError(f'platform link changed: {target}')
    elif target.exists():
        raise RuntimeError(f'refusing to replace {target}')
    else:
        target.symlink_to(source, target_is_directory=True)
print(json.dumps(links, indent=2))
