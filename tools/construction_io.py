"""Bounded-memory records, durable JSONL journals and strict program validation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re


def records(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    try:
        import ijson
    except ImportError:
        ijson = None
    if ijson is not None:
        with path.open('rb') as stream:
            if not stream.read(65536).lstrip().startswith(b'['):
                raise ValueError(f'Expected JSON array: {path}')
            stream.seek(0)
            yield from ijson.items(stream, 'item', use_float=True)
        return
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buffer = stream.read(65536).lstrip()
        if not buffer.startswith('['):
            raise ValueError(f'Expected JSON array or JSONL: {path}')
        buffer = buffer[1:]
        first = True
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(']'):
                if buffer[1:].strip() or stream.read().strip():
                    raise ValueError('Trailing data after JSON array')
                return
            if not first:
                if not buffer:
                    buffer += stream.read(65536)
                if not buffer.startswith(','):
                    raise ValueError('Missing JSON array separator')
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk = stream.read(65536)
                    if not chunk:
                        raise
                    buffer += chunk
            yield row
            first = False
            buffer = buffer[end:]
            if not buffer.strip():
                buffer += stream.read(65536)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def recover_journal(path, metadata_only=False):
    """Recover only a final torn write; reject corruption in earlier records."""
    path = Path(path)
    done = {}
    if not path.exists():
        return done
    with path.open('r+b') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if stream.read(1):
                    raise ValueError(f'Corrupt journal before last record: {path}:{offset}')
                stream.truncate(offset)
                break
            if not line.endswith(b'\n'):
                stream.truncate(offset)
                break
            done[row['sample_id']] = ({key:row.get(key) for key in ['sample_id','status','correct','accepted']} | {'offset':offset}) if metadata_only else row
    return done


def latest_records(path, index):
    with Path(path).open('rb') as stream:
        for item in index.values():
            stream.seek(item['offset'])
            yield json.loads(stream.readline())


def append_record(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def answer_matches(output, row):
    text = str(output).strip()
    options = row.get('options', [])
    if options:
        index = int(row['answer'])
        if text.casefold() == str(options[index]).strip().casefold():
            return True
        match = re.fullmatch(r'\(?([A-Z])\)?[.\s]*', text)
        return bool(match and ord(match[1]) - 65 == index)
    return text.casefold().rstrip('.') == str(row['answer']).strip().casefold().rstrip('.')


TOOLS = {'Count', 'Video_query', 'Query_Actions', 'Query_Objs', 'Filter_frames_with_act',
         'Filter_frames_with_obj', 'trim', 'Find', 'select_answer', 'exist', 'Video_summary'}
BUILTINS = {'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'min': min,
            'max': max, 'sum': sum, 'abs': abs, 'round': round, 'sorted': sorted,
            'range': range, 'enumerate': enumerate, 'zip': zip, 'any': any, 'all': all}


def compile_program(code):
    code = code.strip()
    if code.startswith('```'):
        code = re.sub(r'^```(?:python)?\s*\n', '', code)
        code = re.sub(r'\n```\s*$', '', code)
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Expected one execute_command function')
    function = tree.body[0]
    if function.name != 'execute_command' or function.decorator_list:
        raise ValueError('Invalid generated function')
    if [x.arg for x in function.args.args] != ['video_clip', 'query', 'possible_answers']:
        raise ValueError('Generated function must take video_clip, query, possible_answers')
    if function.args.defaults or function.args.kw_defaults or function.args.kwonlyargs or function.args.vararg or function.args.kwarg or function.args.posonlyargs:
        raise ValueError('Unsupported function signature')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda,
                             ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.With,
                             ast.Try, ast.While, ast.Delete, ast.Raise, ast.Yield,
                             ast.YieldFrom, ast.Await)):
            raise ValueError(f'Unsupported generated operation: {type(node).__name__}')
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ValueError('Nested definitions are unsupported')
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Private names are unsupported')
        if isinstance(node, ast.Attribute) and node.attr not in {'start', 'end', 'num_frames', 'fps', 'clip_len', 'append', 'get', 'items', 'keys', 'values', 'lower', 'upper', 'strip', 'split', 'join', 'count', 'index'}:
            raise ValueError(f'Unsupported generated attribute: {node.attr}')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in TOOLS | set(BUILTINS):
            raise ValueError(f'Unsupported generated call: {node.func.id}')
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(x, ast.Attribute) for target in targets for x in ast.walk(target)):
                raise ValueError('Attribute mutation is unsupported')
    return compile(tree, '<aotd-generated-program>', 'exec'), code
