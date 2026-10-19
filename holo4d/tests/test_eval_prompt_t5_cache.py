from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holo4d.eval_prompt_t5_cache import (
    load_eval_prompt_cache,
    save_eval_prompt_cache,
)


class TestEvalPromptT5Cache(unittest.TestCase):
    def test_roundtrip_and_strict_coverage(self):
        root = Path(tempfile.mkdtemp(prefix="eval_t5_cache_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        source = {
            "caption one": torch.ones(3, 8, dtype=torch.bfloat16),
            "caption two": torch.zeros(5, 8, dtype=torch.bfloat16),
        }
        save_eval_prompt_cache(root, source, provenance={"dataset": "unit"})
        loaded = load_eval_prompt_cache(
            root, expected_prompts=list(source), strict=True
        )
        self.assertEqual(set(loaded), set(source))
        self.assertEqual(tuple(loaded["caption two"].shape), (5, 8))
        with self.assertRaisesRegex(RuntimeError, "misses 1"):
            load_eval_prompt_cache(
                root, expected_prompts=["missing caption"], strict=True
            )


if __name__ == "__main__":
    unittest.main()
