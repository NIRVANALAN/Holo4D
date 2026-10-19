import unittest
from types import SimpleNamespace

import torch

from holo4d_runtime.pipeline.wan_video_track4d360 import WanVideoUnit_TargetImageCondition


class _DropoutOnlyFakePipe:
    def __init__(self):
        self.dit = SimpleNamespace(require_vae_embedding=True)
        self.image_encoder = object()
        self.scheduler = SimpleNamespace(training=True)


class I2VPipelineConditionUnitTest(unittest.TestCase):
    def test_target_dropout_preserves_source_v2v_y(self):
        source = torch.full((1, 16, 2, 2, 3), 5.0)
        target = torch.full_like(source, -3.0)
        full_latents = torch.cat([source, target], dim=-1)

        output = WanVideoUnit_TargetImageCondition().process(
            _DropoutOnlyFakePipe(),
            use_target_image_condition=True,
            target_video=None,
            input_latents=full_latents,
            mode_type="v2v",
            v2v_concat_mode="width",
            i2v_condition_drop_prob=1.0,
        )

        y = output["y"]
        self.assertNotIn("clip_feature", output)
        torch.testing.assert_close(y[:, :4, ..., :3], torch.ones_like(y[:, :4, ..., :3]))
        torch.testing.assert_close(y[:, 4:, ..., :3], source)
        self.assertEqual(int(torch.count_nonzero(y[..., 3:])), 0)

    def test_t2v_source_drop_does_not_condition_on_random_source_latents(self):
        full_latents = torch.randn(1, 16, 2, 2, 6)
        output = WanVideoUnit_TargetImageCondition().process(
            _DropoutOnlyFakePipe(),
            use_target_image_condition=True,
            target_video=None,
            input_latents=full_latents,
            mode_type="t2v",
            v2v_concat_mode="width",
            i2v_condition_drop_prob=1.0,
        )
        self.assertEqual(int(torch.count_nonzero(output["y"])), 0)


if __name__ == "__main__":
    unittest.main()
