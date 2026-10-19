import unittest

import torch

from holo4d.conditions.i2v_target_condition import (
    augment_i2v_image_latents,
    build_source_video_condition,
    expand_i2v_condition_to_v2v_domains,
    generate_long_horizon_chunks,
    plan_chunk_tracking_queries,
    plan_long_horizon_chunks,
    select_long_horizon_chunk,
    stitch_long_horizon_chunks,
)


class I2VTargetConditionTest(unittest.TestCase):
    def test_zero_noise_is_identity(self):
        image = torch.randn(2, 3, 4)
        self.assertIs(augment_i2v_image_latents(image, 0.0), image)

    def test_latent_noise_matches_vista4d_formula(self):
        image = torch.full((1, 2, 3, 4), 2.0)
        noise = torch.full_like(image, -2.0)
        augmented = augment_i2v_image_latents(image, 0.05, noise=noise)
        torch.testing.assert_close(augmented, torch.full_like(image, 1.8))

    def test_source_video_and_target_boundary_conditions_are_both_present(self):
        target_condition = torch.arange(1 * 20 * 2 * 3 * 4, dtype=torch.float32).reshape(
            1, 20, 2, 3, 4
        )
        source_latents = torch.full((1, 16, 2, 3, 4), 7.0)
        source_condition = build_source_video_condition(source_latents)
        full_latents = torch.empty(1, 16, 2, 3, 8)
        expanded = expand_i2v_condition_to_v2v_domains(
            target_condition,
            full_latents,
            source_condition=source_condition,
        )
        self.assertEqual(tuple(expanded.shape), (1, 20, 2, 3, 8))
        self.assertGreater(int(torch.count_nonzero(expanded[..., :4])), 0)
        torch.testing.assert_close(expanded[..., :4], source_condition)
        torch.testing.assert_close(expanded[..., 4:], target_condition)
        torch.testing.assert_close(source_condition[:, :4], torch.ones_like(source_condition[:, :4]))
        torch.testing.assert_close(source_condition[:, 4:], source_latents)

    def test_target_dropout_does_not_drop_source_v2v_condition(self):
        target_condition = torch.zeros(1, 20, 2, 3, 4)
        source_latents = torch.randn(1, 16, 2, 3, 4)
        source_condition = build_source_video_condition(source_latents)
        full_latents = torch.empty(1, 16, 2, 3, 8)
        expanded = expand_i2v_condition_to_v2v_domains(
            target_condition,
            full_latents,
            source_condition=source_condition,
        )
        torch.testing.assert_close(expanded[..., :4], source_condition)
        self.assertEqual(int(torch.count_nonzero(expanded[..., 4:])), 0)

    def test_chunk_plan_overlaps_once_and_pads_only_the_tail(self):
        plan = plan_long_horizon_chunks(100, chunk_frames=49)
        self.assertEqual([window.start for window in plan], [0, 48, 96])
        self.assertEqual([window.valid_frames for window in plan], [49, 49, 4])
        self.assertEqual(plan[1].indices[0], 48)
        self.assertEqual(plan[2].indices[:4], (96, 97, 98, 99))
        self.assertTrue(all(index == 99 for index in plan[2].indices[4:]))

        generated = [
            list(range(0, 49)),
            list(range(48, 97)),
            [96, 97, 98, 99] + [99] * 45,
        ]
        self.assertEqual(stitch_long_horizon_chunks(generated, plan), list(range(100)))

    def test_autoregressive_callback_receives_previous_target_last_frame(self):
        plan = plan_long_horizon_chunks(100, chunk_frames=49)
        conditions = []

        def first(window):
            return list(window.indices)

        def continuation(window, previous_target):
            conditions.append(previous_target)
            return list(window.indices)

        stitched, chunks = generate_long_horizon_chunks(
            plan,
            generate_first=first,
            generate_continuation=continuation,
        )
        self.assertEqual(conditions, [48, 96])
        self.assertEqual(stitched, list(range(100)))
        self.assertEqual(len(chunks), 3)

    def test_each_window_selects_a_new_source_chunk_and_relabels_q2(self):
        plan = plan_long_horizon_chunks(100, chunk_frames=49)
        source_frames = list(range(100))
        tracks_world = torch.arange(2 * 100 * 3).reshape(2, 100, 3)

        source_chunks = [select_long_horizon_chunk(source_frames, chunk) for chunk in plan]
        track_chunks = [
            select_long_horizon_chunk(tracks_world, chunk, frame_dim=-2)
            for chunk in plan
        ]
        query_plans = [plan_chunk_tracking_queries(chunk, num_queries=2) for chunk in plan]

        self.assertEqual(source_chunks[0], list(range(49)))
        self.assertEqual(source_chunks[1], list(range(48, 97)))
        self.assertEqual(source_chunks[2][:4], [96, 97, 98, 99])
        self.assertTrue(all(frame == 99 for frame in source_chunks[2][4:]))
        self.assertEqual(tuple(track_chunks[1].shape), (2, 49, 3))
        torch.testing.assert_close(track_chunks[1][:, 0], tracks_world[:, 48])
        torch.testing.assert_close(track_chunks[1][:, 48], tracks_world[:, 96])
        self.assertEqual(
            [query.local_indices for query in query_plans],
            [(0, 48), (0, 48), (0, 48)],
        )
        self.assertEqual(
            [query.source_indices for query in query_plans],
            [(0, 48), (48, 96), (96, 99)],
        )


if __name__ == "__main__":
    unittest.main()
