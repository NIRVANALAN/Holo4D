import runpy
import sys
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image


def _install_train_script_stubs():
    accelerate_mod = types.ModuleType("accelerate")
    accelerate_mod.DistributedDataParallelKwargs = lambda *args, **kwargs: None
    sys.modules.setdefault("accelerate", accelerate_mod)
    accelerate_utils_mod = types.ModuleType("accelerate.utils")

    class InitProcessGroupKwargs:
        def __init__(self, *args, **kwargs):
            pass

    accelerate_utils_mod.InitProcessGroupKwargs = InitProcessGroupKwargs
    sys.modules["accelerate.utils"] = accelerate_utils_mod

    diffusion_mod = types.ModuleType("holo4d_runtime.diffusion")
    diffusion_mod.__all__ = []
    diffusion_mod.__path__ = []
    sys.modules["holo4d_runtime.diffusion"] = diffusion_mod

    base_pipeline_mod = types.ModuleType("holo4d_runtime.diffusion.base_pipeline")

    class PipelineUnit:
        def __init__(
            self,
            input_params=None,
            output_params=None,
            input_params_posi=None,
            input_params_nega=None,
            onload_model_names=(),
        ):
            self.input_params = input_params
            self.output_params = output_params
            self.input_params_posi = input_params_posi
            self.input_params_nega = input_params_nega
            self.onload_model_names = onload_model_names

    base_pipeline_mod.PipelineUnit = PipelineUnit
    sys.modules["holo4d_runtime.diffusion.base_pipeline"] = base_pipeline_mod

    training_module_mod = types.ModuleType("holo4d_runtime.diffusion.training_module")

    class DiffusionTrainingModule:
        def __init__(self):
            pass

        def transfer_data_to_device(self, inputs, device, dtype):
            return inputs

    training_module_mod.DiffusionTrainingModule = DiffusionTrainingModule
    sys.modules["holo4d_runtime.diffusion.training_module"] = training_module_mod

    runner_mod = types.ModuleType("holo4d_runtime.diffusion.runner")
    runner_mod.launch_training_task = lambda *args, **kwargs: None
    sys.modules["holo4d_runtime.diffusion.runner"] = runner_mod

    logger_mod = types.ModuleType("holo4d_runtime.diffusion.logger")
    logger_mod.ModelLogger = object
    sys.modules["holo4d_runtime.diffusion.logger"] = logger_mod

    logging_mod = types.ModuleType("holo4d_runtime.diffusion.logging")
    logging_mod.create_logger = lambda *args, **kwargs: None
    sys.modules["holo4d_runtime.diffusion.logging"] = logging_mod

    pipeline_mod = types.ModuleType("holo4d_runtime.pipelines.wan_video_track4d360")

    class WanVideoTrack4D360Pipeline:
        pass

    class ModelConfig:
        def __init__(self, *args, **kwargs):
            pass

    pipeline_mod.WanVideoTrack4D360Pipeline = WanVideoTrack4D360Pipeline
    pipeline_mod.ModelConfig = ModelConfig
    sys.modules["holo4d_runtime.pipelines.wan_video_track4d360"] = pipeline_mod

    config_mod = types.ModuleType("holo4d.config")
    config_mod.load_track_config = lambda *args, **kwargs: None
    config_mod.add_track4d360_args = lambda parser: parser
    sys.modules["holo4d.config"] = config_mod


def _load_training_module_class():
    _install_train_script_stubs()
    module_ns = runpy.run_path(
        "scripts/wan2.1_fun/train_track4d360.py",
        run_name="track4d360_train_cond_dropout_contracts",
    )
    return module_ns["WanTrack4D360TrainingModule"]


def _build_minimal_module(module_cls):
    module = object.__new__(module_cls)
    module.training = True
    module.drop_text_prob = 0.0
    module.cond_drop_track_prob = 0.0
    module.cond_drop_dense_prob = 0.0
    module.cond_drop_geo_prob = 0.0
    module.use_track_null_token_fallback = False
    module.use_dense_null_token_fallback = False
    module._latest_log_metrics = {}
    module.pipe = type(
        "DummyPipe",
        (),
        {
            "device": torch.device("cpu"),
            "torch_dtype": torch.float32,
            "units": [],
            "unit_runner": staticmethod(lambda unit, pipe, *inputs: inputs),
        },
    )()
    module.use_gradient_checkpointing = True
    module.use_gradient_checkpointing_offload = False
    module.v2v_concat_mode = "width"
    module.raw_t2v_target_only = False
    module.disable_domain_embedding = False
    module.use_query_frame_impulse_condition = False
    module.future_prediction = False
    module.future_num_latent_frames = 0
    module.future_loss_weight = 1.0
    module.i2v_target_condition = False
    module.i2v_condition_drop_prob = 0.0
    module.i2v_condition_noise_level = 0.0
    module.mode_cfg = {
        "target_track_use_source_stats": False,
        "multimodality_nvs": False,
        "multimodality_tracking": False,
        "multimodality_track_depth": False,
        "channel_concat": True,
        "channel_modalities": ["depth"],
        "num_channel_modalities": 1,
    }
    module.target_depth_log_cap = 14.0
    module.target_depth_norm_type = "log_depth"
    module.target_depth_max_disparity = 5.0
    module.use_dense_branch = True
    module.use_gs_output = False
    module.return_target_tracks = False
    module.return_target_depth = False
    module._force_track_video_loss_mask = None
    module._printed_dense_branch_warning = False
    module._printed_runtime_grad_ckpt = False
    module.use_plucker_camera_control = False
    module.enable_v2v_plucker_camera_control = False
    module.use_perlayer_camera_control = False
    module.camera_injection_mode = "projector"
    module.camera_plucker_encoder_mode = "patchify"
    module.use_plucker_raymap_tokens = False
    module.plucker_use_ray_origin = True
    module.warped_condition_mode = "off"
    module.warped_appearance_fusion = "concat_proj"
    module.warped_geom_only_dense = True
    module.multi_frame_warped_appearance = False
    module.multi_frame_motion_threshold = 0.01
    module.dense_warp_mode = "zbuffer"
    module.dense_warp_motion_threshold = 0.01
    module.dense_hybrid_disable_dynamic_masks = False
    module.use_vista4d_latent_streams = False
    module.vista4d_streams_pointcloud_only = False
    module.use_gen3c_warp_latent_channels = False
    module.fixed_sigma = None
    module.ar_training = False
    module.ar_config = {}
    module.use_gs_feedback_training = False
    module.gs_config = {}
    module.task = "sft:v2v"
    module.task_to_loss = {
        "sft:v2v": lambda pipe, *inputs: (
            torch.tensor(1.0),
            {"loss_metric": torch.tensor(2.0)},
        )
    }
    return module


def _build_batch():
    frame = Image.new("RGB", (32, 24))
    return {
        "prompt": "prompt",
        "video": [frame, frame],
        "target_video": [frame, frame],
        "tracks_world": torch.zeros((4, 2, 3), dtype=torch.float32),
        "target_cam2world": torch.eye(4).unsqueeze(0).repeat(2, 1, 1),
        "target_intrinsics": torch.eye(3),
        "source_depth_video": torch.ones((2, 24, 32), dtype=torch.float32),
        "source_rgb_video": torch.ones((3, 2, 24, 32), dtype=torch.float32),
        "source_depth_valid": torch.ones((2, 24, 32), dtype=torch.float32),
        "source_cam2world": torch.eye(4).unsqueeze(0).repeat(2, 1, 1),
        "source_intrinsics": torch.eye(3),
        "has_depth": True,
        "camera_only_tier": False,
    }


class CondDropoutContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_cls = _load_training_module_class()

    def test_forward_preserves_dropout_metrics_alongside_loss_metrics(self):
        module = _build_minimal_module(self.module_cls)
        module.cond_drop_track_prob = 1.0

        loss = module.forward(_build_batch())
        metrics = module.consume_latest_log_metrics()

        self.assertEqual(float(loss), 1.0)
        self.assertEqual(metrics["cond_dropout/drop_tracks"], 1.0)
        self.assertEqual(metrics["cond_dropout/drop_dense"], 0.0)
        self.assertEqual(float(metrics["loss_metric"]), 2.0)

    def test_dense_dropout_keeps_source_depth_video_for_channel_concat(self):
        module = _build_minimal_module(self.module_cls)
        module.cond_drop_dense_prob = 1.0

        inputs_shared, _, _ = module.get_pipeline_inputs(_build_batch())

        self.assertIn("source_depth_video", inputs_shared)
        self.assertIn("source_rgb_video", inputs_shared)
        self.assertIn("source_depth_valid", inputs_shared)
        self.assertTrue(torch.equal(
            inputs_shared["source_rgb_video"],
            torch.zeros_like(inputs_shared["source_rgb_video"]),
        ))
        self.assertTrue(torch.equal(
            inputs_shared["source_depth_valid"],
            torch.zeros_like(inputs_shared["source_depth_valid"], dtype=torch.bool),
        ))

    def test_camera_only_tier_forces_track_dense_drop_and_zero_query_impulse(self):
        module = _build_minimal_module(self.module_cls)
        module.use_query_frame_impulse_condition = True
        batch = _build_batch()
        batch["camera_only_tier"] = True
        batch["track_query_idx"] = torch.tensor(0)

        inputs_shared, _, _ = module.get_pipeline_inputs(batch)

        self.assertFalse(bool(inputs_shared["has_tracks"]))
        self.assertEqual(float(inputs_shared["depth_loss_mask"]), 0.0)
        self.assertIn("source_rgb_video", inputs_shared)
        self.assertIn("source_depth_valid", inputs_shared)
        self.assertEqual(int(inputs_shared["track_query_idx"].item()), -1)

    def test_graph_signature_records_dense_and_track_paths(self):
        module = _build_minimal_module(self.module_cls)
        module.cond_drop_dense_prob = 1.0
        batch = _build_batch()

        module.get_pipeline_inputs(batch)
        sig = module.consume_latest_graph_signature()

        self.assertEqual(sig["camera_only_tier"], 0)
        self.assertEqual(sig["drop_tracks"], 0)
        self.assertEqual(sig["drop_dense"], 1)
        self.assertEqual(sig["track_path"], 1)
        self.assertEqual(sig["dense_path"], 1)

    def test_null_token_fallback_flags_are_exposed_to_pipeline_inputs(self):
        module = _build_minimal_module(self.module_cls)
        module.use_track_null_token_fallback = True
        module.use_dense_null_token_fallback = True

        inputs_shared, _, _ = module.get_pipeline_inputs(_build_batch())

        self.assertTrue(inputs_shared["use_track_null_token_fallback"])
        self.assertTrue(inputs_shared["use_dense_null_token_fallback"])

    def test_fourrc_source_intrinsics_are_forwarded_to_pipeline_inputs(self):
        module = _build_minimal_module(self.module_cls)
        batch = _build_batch()
        batch["source_intrinsics_4rc"] = torch.eye(3) * 2.0
        batch["source_intrinsics_4rc"][2, 2] = 1.0
        batch["source_intrinsics_gt"] = torch.eye(3) * 3.0
        batch["source_intrinsics_gt"][2, 2] = 1.0

        inputs_shared, _, _ = module.get_pipeline_inputs(batch)

        self.assertTrue(torch.equal(
            inputs_shared["source_intrinsics_4rc"],
            batch["source_intrinsics_4rc"],
        ))
        self.assertTrue(torch.equal(
            inputs_shared["source_intrinsics_gt"],
            batch["source_intrinsics_gt"],
        ))

    def test_shared_args_define_null_token_fallback_flags(self):
        text = open("holo4d/shared_args.py", "r", encoding="utf-8").read()
        self.assertIn("--use_track_null_token_fallback", text)
        self.assertIn("--use_dense_null_token_fallback", text)

    def test_dedicated_identity_seed_uses_step_rank_and_dataset_context(self):
        module = _build_minimal_module(self.module_cls)
        batch = _build_batch()
        batch["dataset_index"] = 37

        with patch.dict(
            "os.environ",
            {"TRACK4D360_DEDICATED_IDENTITY_SAMPLING_SEED": "1"},
            clear=False,
        ):
            module.set_training_step_context(30000, micro_step=0, process_index=1)
            first = module.get_pipeline_inputs(batch)[0]["track_identity_sampling_seed"]
            second = module.get_pipeline_inputs(batch)[0]["track_identity_sampling_seed"]
            module.set_training_step_context(30001, micro_step=0, process_index=1)
            next_step = module.get_pipeline_inputs(batch)[0]["track_identity_sampling_seed"]

        self.assertEqual(first, second)
        self.assertNotEqual(first, next_step)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1 << 63)

    def test_dedicated_identity_seed_is_backward_compatible_when_disabled(self):
        module = _build_minimal_module(self.module_cls)
        with patch.dict(
            "os.environ",
            {"TRACK4D360_DEDICATED_IDENTITY_SAMPLING_SEED": "0"},
            clear=False,
        ):
            inputs_shared, _, _ = module.get_pipeline_inputs(_build_batch())
        self.assertNotIn("track_identity_sampling_seed", inputs_shared)


if __name__ == "__main__":
    unittest.main()
