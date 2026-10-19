import json
import sys
from pathlib import Path

from holo4d import hub


def test_public_hub_names_are_clean():
    assert hub.CHECKPOINT_FILES["t2v"] == "checkpoints/t2v_49_chunk.safetensors"
    assert hub.CHECKPOINT_FILES["i2v"] == "checkpoints/i2v_49_rollout.safetensors"
    assert hub.CHECKPOINT_FILES["5b-commercial"] == (
        "checkpoints_5b/commercial_dense_noid_384x672_step29000.safetensors"
    )
    assert hub.WAN_REPOS["5b"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert len(hub.WAN_REVISIONS["5b"]) == 40
    assert hub.VISTA4D_BUNDLE == "evaluation/vista4d_50_inputs.npz"
    assert hub.DEMO_BUNDLE == "evaluation/syn4d_flying_group_2chunk_demo.npz"
    assert len(hub.WAN_REVISIONS["t2v"]) == 40
    assert len(hub.WAN_REVISIONS["i2v"]) == 40


def test_checkpoint_resolution(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(
        hub, "download_release_file",
        lambda filename, *, root, repo_id=None: calls.append((filename, root)) or tmp_path,
    )
    assert hub.download_checkpoint("t2v", root=tmp_path) == tmp_path
    assert calls == [("checkpoints/t2v_49_chunk.safetensors", tmp_path)]


def test_commercial_5b_checkpoint_resolution(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(
        hub,
        "download_release_file",
        lambda filename, *, root, repo_id=None: calls.append(filename) or tmp_path,
    )
    assert hub.download_checkpoint("5b-commercial", root=tmp_path) == tmp_path
    assert calls == [
        "checkpoints_5b/commercial_dense_noid_384x672_step29000.safetensors"
    ]


def test_checkpoint_cli_preserves_safetensors_snapshot_name(
    monkeypatch, tmp_path: Path, capsys
):
    blob = tmp_path / "blob-without-extension"
    blob.write_bytes(b"payload")
    snapshot = tmp_path / "checkpoints" / "t2v_49_chunk.safetensors"
    snapshot.parent.mkdir()
    snapshot.symlink_to(blob)
    monkeypatch.setattr(hub, "download_checkpoint", lambda kind, *, root: snapshot)
    monkeypatch.setattr(sys, "argv", ["holo4d-download", "checkpoint", "t2v"])

    hub.main()

    printed = Path(capsys.readouterr().out.strip())
    assert printed == snapshot.absolute()
    assert printed.suffix == ".safetensors"
    assert printed.is_file()


def test_wan_resolution_is_revision_pinned(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(tmp_path),
    )
    assert hub.download_wan("i2v", root=tmp_path) == tmp_path
    assert calls[0]["repo_id"] == hub.WAN_REPOS["i2v"]
    assert calls[0]["revision"] == hub.WAN_REVISIONS["i2v"]


def test_valid_vista4d_tree(tmp_path: Path):
    (tmp_path / "vista4d_eval").mkdir()
    (tmp_path / "vista4d_eval" / "metadata.csv").write_text("video,camera\n")
    for name in ("frames", "cow", "vggt"):
        (tmp_path / name).mkdir()
    (tmp_path / "bundle_manifest.json").write_text(json.dumps({"format": hub.FORMAT}))
    assert hub._valid_vista4d_tree(tmp_path)


def test_examples_download_is_pinned_and_does_not_accept_partial_cache(monkeypatch, tmp_path):
    import pytest

    calls = []
    source = tmp_path / "example_download" / hub.EXAMPLE_PREFIX
    source.mkdir(parents=True)
    entries = [{"id": str(i), "condition": f"{i}.npz"} for i in range(10)]
    (source / "manifest.json").write_text(json.dumps({"format": "holo4d-dense-example-v1", "examples": entries}))
    for i in range(10):
        (source / f"{i}.npz").write_bytes(b"fixture")
    monkeypatch.setattr(hub, "snapshot_download", lambda **kw: calls.append(kw))
    manifest = hub.prepare_examples(root=tmp_path)
    assert calls[0]["revision"] == hub.EXAMPLES_REVISION
    assert calls[0]["allow_patterns"] == [f"{hub.EXAMPLE_PREFIX}/*"]
    (manifest.parent / "0.npz").unlink()
    with pytest.raises(ValueError, match="lack a condition"):
        hub.prepare_examples(root=tmp_path)
