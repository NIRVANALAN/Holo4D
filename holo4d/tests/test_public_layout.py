"""Small release tests for the public Holo4D tree.

These checks deliberately avoid importing CUDA/model dependencies.  They make
sure a source checkout contains the public launchers and does not accidentally
include private data or model weights.
"""

from pathlib import Path
import argparse

from holo4d.shared_args import add_source_backend_args


ROOT = Path(__file__).resolve().parents[2]


def test_backend_roots_have_no_cluster_defaults(monkeypatch):
    monkeypatch.delenv("COW_ROOT", raising=False)
    monkeypatch.delenv("VGGT_ROOT", raising=False)
    parser = argparse.ArgumentParser()
    add_source_backend_args(parser)
    args = parser.parse_args([])
    assert args.source_backend == "4rc"
    assert args.cow_root is None and args.vggt_root is None


def test_backend_roots_accept_environment_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("COW_ROOT", "/data/cow")
    monkeypatch.setenv("VGGT_ROOT", "/data/vggt")
    parser = argparse.ArgumentParser()
    add_source_backend_args(parser)
    args = parser.parse_args([])
    assert (args.cow_root, args.vggt_root) == ("/data/cow", "/data/vggt")
    args = parser.parse_args(["--cow_root", "/custom/cow", "--vggt_root", "/custom/vggt"])
    assert (args.cow_root, args.vggt_root) == ("/custom/cow", "/custom/vggt")


def test_public_layout_and_namespace() -> None:
    assert (ROOT / "holo4d" / "__init__.py").is_file()
    assert not (ROOT / "track4d_360").exists()
    assert (ROOT / "inference" / "sample_19500_syn4d.sh").is_file()
    assert (ROOT / "inference" / "sample_19500_davis.sh").is_file()


def test_public_evaluation_entrypoints() -> None:
    expected = (
        ROOT / "eval" / "iphone" / "eval_iphone_nvs.py",
        ROOT / "eval" / "davis" / "eval_track4d360_davis_4rc_novel_traj.py",
        ROOT / "eval" / "syn4d" / "eval_flying_group_4rc_depth_gt.py",
        ROOT / "eval" / "common" / "aggregate_metrics.py",
    )
    assert all(path.is_file() for path in expected)


def test_release_does_not_embed_private_assets() -> None:
    assert not (ROOT / "weights").exists()
    assert not (ROOT / "datasets").exists()
    assert not any(ROOT.rglob("*.safetensors"))
