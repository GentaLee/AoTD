"""Exercise the actual DeepSeek formatting method without loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class ProgramPrefixTests(unittest.TestCase):
    def test_filter_receives_trace_and_proposed_answer_without_label(self):
        from build_cot import TextBackend, AGENT
        calls = []
        backend = TextBackend.__new__(TextBackend)
        backend.stage = 'filter'
        backend.prompt = (AGENT/'prompts/cot_filter.prompt').read_text()
        backend.tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kw: messages[-1]['content'],
            eos_token_id=2, convert_tokens_to_ids=lambda _: 3)
        backend.pipe = lambda prompt, **kw: calls.append(prompt) or [{'generated_text': 'False'}]
        row = dict(question='When?', options=[], answer='PRIVATE_LABEL_SENTINEL',
                   program_output='After.', execution_trace='Observed before/after evidence.',
                   rationale='It happened before.')
        self.assertFalse(backend(row, None, None)['accepted'])
        self.assertIn('Observed before/after evidence.', calls[0])
        self.assertIn('After.', calls[0])
        self.assertIn('It happened before.', calls[0])
        self.assertNotIn('PRIVATE_LABEL_SENTINEL', calls[0])
        self.assertNotIn('INSERT_', calls[0])

    def test_prefix_is_in_model_input_and_preserved_in_program(self):
        tree = ast.parse((Path(__file__).resolve().parents[1]/"agent_system/vision_models.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSeekModel")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_")
        module = ast.Module(body=[method], type_ignores=[])
        prefix = "def execute_command(video_clip, query, possible_answers):\n"
        namespace = {"config": SimpleNamespace(deepseek={"program_prefix": prefix})}
        exec(compile(module, "actual_deepseek_forward", "exec"), namespace)
        calls = []
        def generate(text, **kwargs):
            calls.append(text)
            return [{"generated_text": "    return 'yes'"}]
        model = SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=lambda *a, **k: "CHAT\n"), pipeline=generate)
        result = namespace["forward_"](model, "prompt", input_type="video_clip, query, possible_answers")
        self.assertEqual(calls, ["CHAT\n"+prefix])
        self.assertTrue(result.startswith("def execute_command("))
        self.assertTrue(result.endswith("    return 'yes'"))
        self.assertEqual(result.count("def execute_command("), 1)

    def test_video_summary_uses_question_and_preserves_generic_fallback(self):
        tree = ast.parse((Path(__file__).resolve().parents[1]/"agent_system/vision_models.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLaVAVideoModel")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "summary")
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_summary", "exec"), namespace)
        calls = []
        model = SimpleNamespace(inference=lambda clip, prompt: calls.append(prompt) or "observed")
        self.assertEqual(namespace["summary"](model, "frames", "What happened before sitting?"), "observed")
        self.assertIn("What happened before sitting?", calls[-1])
        self.assertIn("actually visible", calls[-1])
        namespace["summary"](model, "frames")
        self.assertIn("one sentence description", calls[-1])
