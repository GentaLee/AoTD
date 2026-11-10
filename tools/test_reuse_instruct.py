import json
from pathlib import Path
import tempfile
import unittest
from reuse_instruct import semantic_data


class ReuseTests(unittest.TestCase):
    def test_relocated_identical_media_pass_but_content_and_order_changes_do_not(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); media=p/"video.mp4"; media.write_bytes(b"fixture")
            alias=p/"alias.mp4"; alias.symlink_to(media)
            row=dict(sample_id="one",video_path=str(media),conversation=["answer"])
            for name,rows in [("a",[row,dict(row,sample_id="two")]),("b",[dict(row,video_path=str(alias)),dict(row,sample_id="two",video_path=str(alias))])]:
                (p/(name+".json")).write_text(json.dumps(rows))
                (p/(name+".yaml")).write_text("datasets:\n  - json_path: "+name+".json\n")
            self.assertEqual(semantic_data(p/"a.yaml"),semantic_data(p/"b.yaml"))
            (p/"b.json").write_text(json.dumps([dict(row,sample_id="two"),row]))
            self.assertNotEqual(semantic_data(p/"a.yaml"),semantic_data(p/"b.yaml"))
            (p/"b.json").write_text(json.dumps([dict(row,conversation=["different"]),dict(row,sample_id="two")]))
            self.assertNotEqual(semantic_data(p/"a.yaml"),semantic_data(p/"b.yaml"))
