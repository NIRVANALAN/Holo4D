import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_public_readme_uses_only_release_naming() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "ab12" not in readme.lower()
    assert "/vgg/" not in readme
    assert "/lus/" not in readme
    assert "checkpoints/t2v_49_chunk.safetensors" in readme
    assert "checkpoints/i2v_49_rollout.safetensors" in readme


def test_release_entrypoints_exist_and_match_canonical_contract() -> None:
    vista = (ROOT / "inference/sample_vista4d.sh").read_text()
    long = (ROOT / "inference/sample_mixed_t2v_i2v_syn4d.sh").read_text()
    train = (ROOT / "train/train_384p.sh").read_text()
    train_i2v = (ROOT / "train/train_i2v_384p.sh").read_text()
    demo = (ROOT / "inference/demo_vista4d.sh").read_text()
    demo_t2v = (ROOT / "inference/demo_t2v_49.sh").read_text()
    demo_i2v = (ROOT / "inference/demo_i2v_97.sh").read_text()
    assert "inference/sample_vista4d.sh" in demo
    assert r"\${" not in demo
    assert "holo4d.hub demo" in demo_t2v
    assert "inference/sample_19500_syn4d.sh" in demo_t2v
    assert "holo4d.hub demo" in demo_i2v
    assert "inference/sample_mixed_t2v_i2v_syn4d.sh" in demo_i2v
    assert "FIXED_SOURCE_SEQUENCES=seq_000000" in demo_i2v
    assert "holo4d.hub wan t2v" in vista
    assert "holo4d.hub checkpoint t2v" in vista
    assert "holo4d.hub vista4d" in vista
    assert "holo4d.hub wan t2v" in long
    assert "holo4d.hub wan i2v" in long
    assert "holo4d.hub checkpoint t2v" in long
    assert "holo4d.hub checkpoint i2v" in long
    assert "holo4d.hub wan t2v" in train
    assert "holo4d.hub wan i2v" in train_i2v
    for token in (
        "--vista4d_target_scale_align depth",
        '--trajectory_anchor_mode "${ANCHOR}"',
        "--trajectory_damping",
        "--cow_vggt_scale_align none",
        "--num_inference_steps \"${NUM_INFERENCE_STEPS:-50}\"",
    ):
        assert token in vista
    assert 'native) ANCHOR=vista4d_frame0; DAMPING=1.0' in vista
    assert 'shared-first-frame-half) ANCHOR=source_frame0; DAMPING=0.5' in vista
    assert "--i2v_seed_checkpoint_path" in long
    assert '--i2v_resume_stage "${I2V_RESUME_STAGE:-both}"' in long
    assert "--i2v_continuation_checkpoint_path" in long
    assert "${NUM_INFERENCE_STEPS:-50}" in long
    assert "${CFG_SCALE:-5.0}" in long
    assert "holo4d.tools.i2v_continuation_long_video" in long
    for token in (
        "--task sft:v2v",
        "--model_variant vt2v",
        "--v2v_concat_mode width",
        "--height \"${HEIGHT:-384}\"",
        "--width \"${WIDTH:-672}\"",
        "--num_frames 49",
        "--train_sigma_shift 5.0",
    ):
        assert token in train
    for token in (
        "Wan-AI/Wan2.1-I2V-14B-480P",
        "--i2v_target_condition",
        "${I2V_CONDITION_DROP_PROB:-0.3}",
        "${I2V_CONDITION_NOISE_LEVEL:-0.05}",
        "--train_sigma_shift 5.0",
    ):
        assert token in train_i2v
    for launcher in (vista, long, train, train_i2v):
        assert "SLURM_SUBMIT_DIR" in launcher


def test_training_recipes_have_no_private_path_defaults() -> None:
    for name in ("train_384p.yaml", "train_i2v_384p.yaml"):
        recipe = (ROOT / "config/holo4d" / name).read_text()
        assert "resolution: [672, 384]" in recipe
        assert "caption_dir='${syn4d_caption_dir}'" in recipe
        assert "/vgg/" not in recipe
        assert "/lus/" not in recipe


def test_training_data_doc_matches_canonical_recipe_weights() -> None:
    def logical_weights(name: str) -> list[int]:
        recipe = (ROOT / "config/holo4d" / name).read_text(encoding="utf-8")
        return [
            int(value)
            for value in re.findall(r'^\s*-\s*"([0-9]+) @', recipe, re.MULTILINE)
        ]

    v2v_weights = [52500, 22500, 100000, 50000, 25000]
    i2v_weights = [62500, 39375, 16875, 75000, 37500, 18750]
    assert logical_weights("train_384p.yaml") == v2v_weights
    assert logical_weights("train_i2v_384p.yaml") == i2v_weights
    assert sum(v2v_weights) == 250000
    assert sum(i2v_weights) == 250000

    data_doc = (ROOT / "doc/data.md").read_text(encoding="utf-8")
    compact_doc = data_doc.replace(",", "")
    for weight in v2v_weights + i2v_weights:
        assert f"| {weight} |" in compact_doc
    for token in (
        "70% dynamic / 30% static",
        "77.5% dynamic / 22.5% static",
        "https://huggingface.co/datasets/Syn4D/Syn4D",
        "https://huggingface.co/datasets/KlingTeam/MultiCamVideo-Dataset",
        "https://huggingface.co/datasets/InternRobotics/OmniWorld",
        "https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P",
        "https://huggingface.co/datasets/phuang17/MVS-Synth",
    ):
        assert token in data_doc
    assert "](doc/data.md)" in (
        ROOT / "README.md"
    ).read_text(encoding="utf-8")


def test_duplicate_trainer_entrypoints_do_not_drift() -> None:
    trainer = (ROOT / "train/train_track4d360.py").read_text()
    assert trainer.encode() == (
        ROOT / "scripts/wan2.1_fun/train_track4d360.py"
    ).read_bytes()
    assert 'get("HOLO4D_OFFLOAD_T5_TO_CPU"' in trainer
    assert 'get("HOLO4D_KEEP_T5"' in trainer
    assert "camera-control base_" not in trainer

def test_public_evaluation_entrypoint_is_portable() -> None:
    benchmark = (ROOT / "eval/common/benchmark_env.sh").read_text()
    video_eval = (ROOT / "eval/common/eval_video_metrics.sh").read_text()
    quality = (ROOT / "eval/common/eval_video_quality.sh").read_text()
    canonical = benchmark + video_eval + quality
    for private_root in ("/vgg/", "/lus/", "/scratch/", "/projects/"):
        assert private_root not in canonical
    assert (ROOT / "eval/build_flat_tree.py").is_file()
    assert '${SCRIPT_DIR}/video_quality' in video_eval
    assert 'REPO_ROOT="${SCRIPT_DIR}/video_quality"' in quality
    assert "CONDA_ENV_camera-control" not in video_eval
    assert "HOLO4D_CONDA_ENV" in video_eval
    assert "holo4d_activate_conda" in benchmark
    assert "outputs/benchmarks" in benchmark

    vista_python = (
        ROOT / "examples/wan2.1_fun/eval_track4d360_vista4d_official.py"
    ).read_text()
    for private_root in ("/vgg/", "/lus/", "/scratch/", "/projects/"):
        assert private_root not in vista_python
    assert 'os.environ.get("VISTA4D_EVAL_ROOT")' in vista_python

def test_long_rollout_supports_continuation_only_resume() -> None:
    evaluator = (
        ROOT / "examples/wan2.1_fun/eval_track4d360_hybrid_dense_syn4d.py"
    ).read_text()
    assert '"--i2v_resume_stage"' in evaluator
    assert 'args.i2v_resume_stage == "continuation"' in evaluator
    assert "resuming from completed seed with I2V continuation" in evaluator
