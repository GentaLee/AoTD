"""Shared source selection and full-pool acceptance checks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_protocol(path=None):
    config = json.loads(Path(path or ROOT/'config/construction/sources.json').read_text())
    names = [source['dataset'] for source in config['sources']]
    if not names or len(set(names)) != len(names):
        raise ValueError('Source manifest must contain unique dataset names')
    if set(names) & set(config.get('excluded_sources', [])):
        raise ValueError('An excluded source is enabled in the source manifest')
    return config


def coverage_errors(counts, config):
    """Check each source independently, so a large source cannot hide a missing one."""
    expected = {source['dataset']: source for source in config['sources']}
    errors = []
    extra = set(counts)-set(expected)
    if extra:
        errors.append('Unexpected/excluded training datasets: '+', '.join(sorted(extra)))
    for name, source in expected.items():
        minimum = source['minimum_qa_records']
        if counts.get(name, 0) < minimum:
            errors.append(f'{name}: {counts.get(name, 0)} QA; full pool requires at least {minimum}')
    return errors
