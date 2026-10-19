#!/usr/bin/env bash
# Release-focused tests for the public 384p train/inference/evaluation contract.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export DISABLE_FLASH_ATTN_3="${DISABLE_FLASH_ATTN_3:-1}"
cd "${ROOT}"
exec "${PYTHON_BIN}" -m pytest -q \
  holo4d/tests/test_source_geometry_release.py \
  holo4d/tests/test_robot_rollouts.py \
  holo4d/tests/test_public_paper_evaluation.py \
  holo4d/tests/test_examples_io.py \
  holo4d/tests/test_pipeline_integration.py \
  holo4d/tests/test_davis_4rc_gmos_track_fusion.py \
  holo4d/tests/test_persistent_trajectory_identity.py \
  holo4d/tests/test_release_384p_contract.py \
  holo4d/tests/test_hub.py \
  holo4d/tests/test_demo_rollout_bundle.py \
  holo4d/tests/test_public_layout.py \
  holo4d/tests/test_readme_visuals.py \
  holo4d/tests/test_public_validation_assets.py \
  holo4d/tests/test_vista4d_shared_first_frame.py \
  holo4d/tests/test_vista4d_same_first_frame_alignment.py \
  holo4d/tests/test_validate_vista4d_outputs.py \
  holo4d/tests/test_i2v_target_condition.py \
  holo4d/tests/test_i2v_pipeline_condition_unit.py \
  holo4d/tests/test_i2v_continuation_long_video.py \
  holo4d/tests/test_object_centric_orbit.py \
  holo4d/tests/test_eval_prompt_t5_cache.py \
  holo4d/tests/test_atomic_npz.py \
  holo4d/tests/test_train_track4d360_cond_dropout.py \
  holo4d/tests/test_dl3dv_camera_control_script.py \
  holo4d/tests/test_morton_bit_calibration.py \
  holo4d/tests/test_recammaster_cow_vggt_random_samples.py \
  holo4d/tests/test_raw_video_preprocess_contract.py \
  holo4d/tests/test_syn4d_nvs_compact_tracking_index.py \
  holo4d/tests/test_raw_condition_visualization.py \
  holo4d/tests/test_cowtracker_compat.py \
  holo4d/tests/test_sampling_audit_contracts.py \
  holo4d/tests/test_checkpoint_coverage.py \
  holo4d/tests/test_rollout_coordinates.py \
  holo4d/tests/test_release_dependency_groups.py \
  holo4d/tests/test_rollout_readme_gifs.py \
  holo4d/tests/test_training_launcher_flags.py
