"""Real processor regressions for generated object descriptions; no weights needed.

Set AOTD_OWLV2_MODEL_DIR to the offline detector directory. The isolated forward
method avoids importing unrelated teachers or loading any model weights.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import unittest
from unittest.mock import Mock


MODEL_DIR = os.environ.get('AOTD_OWLV2_MODEL_DIR')


@unittest.skipUnless(MODEL_DIR, 'Set AOTD_OWLV2_MODEL_DIR for offline processor regressions')
class Owlv2TextBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Owlv2Config, Owlv2Processor
        cls.torch = torch
        cls.processor = Owlv2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
        cls.config = Owlv2Config.from_pretrained(MODEL_DIR, local_files_only=True)
        source = Path(__file__).resolve().parents[1]/'agent_system/vision_models.py'
        module = ast.parse(source.read_text())
        detector = next(node for node in module.body
                        if isinstance(node, ast.ClassDef) and node.name == 'Owlv2Model')
        forward = next(node for node in detector.body
                       if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        namespace = dict(torch=torch, Union=Union)
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), namespace)
        cls.forward = staticmethod(namespace['forward'])

    def setUp(self):
        torch = self.torch
        self.model = Mock()
        self.model.device = torch.device('cpu')
        self.model.config = self.config
        self.processor.post_process_object_detection = Mock(return_value=[dict(
            boxes=torch.empty((0, 4)), scores=torch.empty(0), labels=torch.empty(0, dtype=torch.long))])
        self.adapter = SimpleNamespace(model=self.model, processor=self.processor)
        self.image = torch.zeros((3, 32, 32), dtype=torch.uint8)

    def encoded(self, objects):
        self.forward(self.adapter, self.image, objects)
        return self.model.call_args.kwargs

    def test_long_generated_description_respects_position_limit_and_keeps_eos(self):
        text = 'two different banners with many separate written messages ' * 12
        unbounded = self.processor.tokenizer('a photo of a '+text, truncation=False)
        limit = self.config.text_config.max_position_embeddings
        self.assertGreater(len(unbounded['input_ids']), limit)
        inputs = self.encoded(text)
        self.assertEqual(tuple(inputs['input_ids'].shape), (1, limit))
        self.assertEqual(inputs['input_ids'][0, -1].item(), self.processor.tokenizer.eos_token_id)

    def test_multiple_mixed_length_objects_share_the_loaded_model_limit(self):
        objects = ['person', 'a long object description ' * 40, 'banner']
        inputs = self.encoded(objects)
        self.assertEqual(tuple(inputs['input_ids'].shape),
                         (len(objects), self.config.text_config.max_position_embeddings))
        self.assertEqual(inputs['attention_mask'].shape, inputs['input_ids'].shape)
        expected = self.processor.tokenizer('a photo of a person', padding='max_length',
                                            max_length=self.config.text_config.max_position_embeddings,
                                            return_tensors='pt')
        self.assertTrue(self.torch.equal(inputs['input_ids'][0], expected['input_ids'][0]))

    def test_short_query_preserves_existing_processor_encoding(self):
        expected = self.processor(text=['a photo of a person'], images=self.image,
                                  return_tensors='pt')
        actual = self.encoded('person')
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertTrue(self.torch.equal(actual[key], expected[key]), key)

    def test_loaded_model_limit_wins_over_an_incorrect_tokenizer_default(self):
        previous = self.processor.tokenizer.model_max_length
        try:
            self.processor.tokenizer.model_max_length = 128
            inputs = self.encoded('large complicated object description ' * 40)
        finally:
            self.processor.tokenizer.model_max_length = previous
        self.assertEqual(inputs['input_ids'].shape[1],
                         self.config.text_config.max_position_embeddings)

    def test_model_runtime_errors_propagate_without_reclassification_or_retry(self):
        error = RuntimeError('model backend failure fixture')
        self.model.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.encoded('banner')
        self.assertIs(caught.exception, error)
        self.assertEqual(self.model.call_count, 1)
        self.processor.post_process_object_detection.assert_not_called()


if __name__ == '__main__':
    unittest.main()
