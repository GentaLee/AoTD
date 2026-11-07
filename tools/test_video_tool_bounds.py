"""Check that invalid temporal/spatial selections fail before teacher inference."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys

import torch


def load_video_tools():
    # Import the real tools without loading the five teacher model weights.
    process = ModuleType('video_process')
    process.forward = Mock(side_effect=AssertionError('Unexpected teacher inference'))
    configs = ModuleType('configs')
    configs.config = SimpleNamespace(cot={'save_cot_dir': ''}, crop_larger_margin=True)
    path = Path(__file__).resolve().parents[1] / 'agent_system/video_clip.py'
    spec = importlib.util.spec_from_file_location('bounded_video_tools', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'video_process': process, 'configs': configs}):
        spec.loader.exec_module(module)
    return module


class VideoToolBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = load_video_tools()

    def setUp(self):
        self.frames = torch.zeros((4, 24, 32, 3), dtype=torch.uint8)

    def test_nonempty_temporal_selection_preserves_absolute_frame_offsets(self):
        clip = self.tools.VideoClip(self.frames, start=10, end=14, parent_start=0)
        selected = self.tools.trim(clip, start=11, end=13)
        self.assertEqual((selected.start, selected.end, selected.num_frames), (11, 13, 2))
        self.assertEqual(tuple(selected.video.shape), (2, 24, 32, 3))

    def test_empty_or_inverted_selection_is_rejected_before_teacher_call(self):
        clip = self.tools.VideoClip(self.frames)
        with patch.object(self.tools, 'forward_') as inference:
            for start, end in [(3, 1), (8, 12), (-12, -8)]:
                with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                    selected = self.tools.trim(clip, start=start, end=end)
                    self.tools.Video_query(selected, 'What happened?')
            inference.assert_not_called()

    def test_malformed_or_empty_frames_are_rejected(self):
        for frames in [torch.empty((0, 24, 32, 3)), torch.empty((1, 0, 32, 3)),
                       torch.empty((1, 24, 32, 1)), torch.empty((3, 24, 32)), []]:
            with self.subTest(shape=getattr(frames, 'shape', None)), self.assertRaises(ValueError):
                self.tools.VideoClip(frames)

    def test_invalid_boxes_are_rejected_before_resize(self):
        frame = self.frames[0].permute(2, 0, 1)
        for box in [[100, 100, 120, 120], [-100, -100, -80, -80],
                    [8, 4, 4, 12], [4, 8, 12, 8], [0, 0, float('nan'), 12],
                    [0, 0, float('inf'), 12]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.tools.crop(frame, torch.tensor(box))

    def test_partially_visible_box_is_clamped_with_or_without_margin(self):
        frame = self.frames[0].permute(2, 0, 1)
        for margin in [True, False]:
            with patch.object(self.tools.config, 'crop_larger_margin', margin):
                result = self.tools.crop(frame, torch.tensor([-4, -2, 16, 12]))
                self.assertEqual(tuple(result.shape), (3, 224, 224))

    def test_find_normalizes_different_detection_sizes_for_video_teacher(self):
        clip = self.tools.VideoClip(self.frames)
        boxes = [torch.tensor([[1, 2, 10 + i, 12 + i]]) for i in range(4)]
        with patch.object(self.tools, 'forward_', side_effect=boxes):
            selected = self.tools.Find(clip, 'person')
        self.assertEqual(tuple(selected.video.shape), (4, 224, 224, 3))

    def test_actual_backend_errors_are_not_swallowed(self):
        clip = self.tools.VideoClip(self.frames)
        failure = RuntimeError('Backend allocation failed')
        with patch.object(self.tools, 'forward_', side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                self.tools.Video_query(clip, 'What happened?')
        self.assertIs(caught.exception, failure)


if __name__ == '__main__':
    unittest.main()
