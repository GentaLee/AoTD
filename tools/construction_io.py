"""Bounded-memory records, durable JSONL journals and strict program validation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re


class GeneratedProgramError(ValueError):
    """A Python error originating in the generated program, not a model/tool."""


CODE_KEYS = ('driver_sha256', 'io_sha256', 'tool_sources_sha256')


def tool_sources_sha256(agent):
    """Bind the executed tools and their local prompts, not only the driver."""
    agent = Path(agent)
    paths = sorted({*agent.rglob('*.py'), *agent.rglob('*.prompt'),
                    *agent.glob('configs/*.yaml')})
    return {str(path.relative_to(agent)): sha256(path) for path in paths
            if '__pycache__' not in path.parts}


def execute_program(compiled, namespace, video, query, answers):
    try:
        exec(compiled, namespace)
        return namespace['execute_command'](video, query, answers)
    except (NameError, ArithmeticError, LookupError, AttributeError, TypeError,
            ValueError, AssertionError, StopIteration) as error:
        tail = error.__traceback__
        while tail.tb_next is not None:
            tail = tail.tb_next
        if tail.tb_frame.f_code.co_filename != '<aotd-generated-program>':
            raise
        raise GeneratedProgramError(f'{type(error).__name__}: {error}') from error


def reviewed_code_migration(previous, current, policy_path):
    """Permit only an exact reviewed code pair with every other input unchanged."""
    policy_path = Path(policy_path)
    if not policy_path.is_file():
        return None
    keys = CODE_KEYS
    mutable = {*keys, 'input_sha256', 'input_bytes', 'input_revisions', 'code_migrations'}
    fixed = lambda value: {key: item for key, item in value.items() if key not in mutable}
    if fixed(previous) != fixed(current):
        return None
    for item in json.loads(policy_path.read_text()).get('migrations', []):
        if (all(previous.get(key) == item['from'].get(key) for key in keys)
                and all(current.get(key) == item['to'].get(key) for key in keys)):
            return dict(migration_id=item['migration_id'], previous_code=item['from'],
                        current_code=item['to'], policy_sha256=sha256(policy_path),
                        reason=item['reason'])
    return None


def migrate_generated_failures(journal, index, migration):
    """Append corrected classifications; preserve every original journal byte."""
    updates = []
    for row in latest_records(journal, index):
        if row.get('status') != 'error':
            continue
        frames = re.findall(r'File "([^\"]+)"', row.get('traceback', ''))
        kind = row.get('error', '').split(':', 1)[0]
        if (not frames or frames[-1] != '<aotd-generated-program>'
                or kind not in {'NameError', 'UnboundLocalError', 'ZeroDivisionError',
                                'OverflowError', 'AssertionError', 'StopIteration'}):
            continue
        updated = dict(row, status='rejected', rejection_kind='generated_program',
                       classification_migration=migration['migration_id'], previous_status='error')
        updates.append(updated)
    if updates:
        with Path(journal).open('a') as stream:
            for row in updates:
                append_record(stream, row)
    return len(updates)


def should_process(saved, retry_rejected=False):
    # Infrastructure failures always retry on restart; rejected data is terminal
    # unless an explicit retry is requested. Successful records never repeat.
    return saved is None or saved.get('status') == 'error' or (
        retry_rejected and saved.get('status') == 'rejected')


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
    normalized = text.casefold().rstrip('.')
    expected = str(row['answer']).strip().casefold().rstrip('.')
    # AGQA binary labels use yes/no while the visual API can return bool.
    # Only normalize a complete boolean token; no substring/prose matching,
    # no mapping for object/action answers or multiple-choice indices.
    if row.get('dataset') == 'AGQA' and expected in {'yes', 'no'}:
        normalized = {'true': 'yes', 'false': 'no'}.get(normalized, normalized)
    return normalized == expected


def trace_question_context(row):
    """Expose choices and the program-selected text, never the ground-truth label."""
    question = row['question']
    options = row.get('options') or []
    if not options:
        return question
    choices = '\n'.join(f'({chr(65+i)}) {option}' for i, option in enumerate(options))
    output = str(row.get('program_output', '')).strip()
    selected = output
    match = re.fullmatch(r'\(?([A-Z])\)?[.\s]*', output)
    if match and 0 <= ord(match[1])-65 < len(options):
        selected = str(options[ord(match[1])-65])
    return f'{question}\nAnswer choices:\n{choices}\nRecorded program answer: {output}\nRecorded answer text: {selected}'


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
