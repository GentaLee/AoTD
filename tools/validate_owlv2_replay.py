"""Replay a recorded construction query through the real offline OWLv2 detector.

Run under the normal workload/keep-alive guard. This command loads real model
weights and a frame from the normalized source video; it creates no QA labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from construction_io import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--construction-dir', type=Path,
                        default=ROOT/'data/construction/NExT-QA')
    parser.add_argument('--sample-id', default='4563040340_0')
    parser.add_argument('--query-file', type=Path,
                        help='Preserved original object query; otherwise use the saved Count trace')
    parser.add_argument('--expected-tokens', type=int,
                        help='Assert the original query token count, for example 97')
    parser.add_argument('--output', type=Path, required=True,
                        help='Persistent JSON evidence output')
    return parser.parse_args()


def replay(args):
    source = args.construction_dir.resolve()
    config = source/'agent_config.yaml'
    if not config.is_file():
        raise FileNotFoundError('Construction agent_config.yaml is required')
    os.environ['AOTD_AGENT_CONFIG'] = str(config)
    os.environ['AOTD_AGENT_STRICT'] = '1'
    sys.path.insert(0, str(ROOT/'agent_system'))

    import decord
    import numpy as np
    import torch
    import transformers
    from vision_models import Owlv2Model

    if not torch.cuda.is_available():
        raise RuntimeError('A guarded CUDA device is required for real model replay')
    if args.query_file:
        query = args.query_file.read_text().strip()
    else:
        trace = source/'traces'/(hashlib.sha256(args.sample_id.encode()).hexdigest()+'.txt')
        queries = [line.removeprefix('counting ') for line in trace.read_text().splitlines()
                   if line.startswith('counting ')]
        if not queries:
            raise ValueError('The selected construction trace contains no Count object query')
        query = queries[-1]
    if not query:
        raise ValueError('The recorded object query is empty')
    row = None
    with (source/'qa_normalized.jsonl').open() as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate['sample_id'] == args.sample_id:
                if row is not None:
                    raise ValueError('The normalized source has duplicate sample IDs')
                row = candidate
    if row is None:
        raise ValueError('The selected sample is absent from the normalized source')
    reader = decord.VideoReader(row['video_path'], num_threads=1, ctx=decord.cpu(0))
    if not len(reader):
        raise ValueError('The original source video has no decodable frames')
    indices = np.linspace(0, len(reader), 32, endpoint=False).astype(int)
    # Text-encoder bounds are independent of the image. Use an actual frame
    # from the same 32-frame sampling schedule as construction, not a fixture.
    frame = torch.from_numpy(reader.get_batch(indices[1:2]).asnumpy()[0]).permute(2, 0, 1)
    started = time.monotonic()
    detector = Owlv2Model(0)
    limit = detector.model.config.text_config.max_position_embeddings
    baseline = detector.processor.tokenizer('a photo of a '+query, truncation=False)
    original_tokens = len(baseline['input_ids'])
    if args.expected_tokens is not None and original_tokens != args.expected_tokens:
        raise AssertionError(f'Original input has {original_tokens} tokens; expected {args.expected_tokens}')
    captured = {}

    def inspect_input(_module, _args, kwargs):
        ids = kwargs['input_ids']
        mask = kwargs['attention_mask']
        captured['actual_input_tokens'] = ids.shape[-1]
        # The final attended token is EOS for both truncated and padded inputs.
        eos_positions = mask.sum(dim=-1).long()-1
        ends = ids.gather(1, eos_positions[:, None]).flatten()
        captured['eos_preserved'] = bool((ends == detector.processor.tokenizer.eos_token_id).all().item())

    def inspect_output(_module, _args, output):
        captured['logits_shape'] = list(output.logits.shape)
        captured['boxes_shape'] = list(output.pred_boxes.shape)
        captured['finite_logits'] = bool(torch.isfinite(output.logits).all().item())
        captured['finite_boxes'] = bool(torch.isfinite(output.pred_boxes).all().item())

    before = detector.model.register_forward_pre_hook(inspect_input, with_kwargs=True)
    after = detector.model.register_forward_hook(inspect_output)
    try:
        boxes = detector.forward(frame, query)
        torch.cuda.synchronize()
    finally:
        before.remove()
        after.remove()
    result = dict(sample_id=args.sample_id, query_sha256=hashlib.sha256(query.encode()).hexdigest(),
                  original_input_tokens=original_tokens, model_text_limit=limit,
                  returned_boxes=len(boxes), seconds=round(time.monotonic()-started, 3),
                  software=dict(torch=torch.__version__, transformers=transformers.__version__,
                                cuda=torch.version.cuda), **captured)
    if captured.get('actual_input_tokens') != limit:
        raise AssertionError('Detector inputs did not respect the loaded text context limit')
    if not all(captured.get(key) for key in ['eos_preserved', 'finite_logits', 'finite_boxes']):
        raise AssertionError('Detector EOS or finite-output validation failed')
    return dict(result, status='PASS')


def main():
    args = arguments()
    try:
        result = replay(args)
    except Exception as error:
        atomic_json(args.output, dict(sample_id=args.sample_id, status='FAIL',
                                      error_type=type(error).__name__))
        raise
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
