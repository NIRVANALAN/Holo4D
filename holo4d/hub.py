"""Download and resolve the public Holo4D inference assets."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from holo4d.tools.vista4d_release_bundle import FORMAT, extract, parser as bundle_parser

RELEASE_REPO = "yslan/Holo4D-2026"
CHECKPOINT_FILES = {
    "t2v": "checkpoints/t2v_49_chunk.safetensors",
    "i2v": "checkpoints/i2v_49_rollout.safetensors",
    # Commercial-mixture Wan 2.2 TI2V-5B release.  This is deliberately a
    # separate key from the 14B ``t2v``/``i2v`` demos: it is a 384p,
    # no-identity first-window checkpoint and must not be mixed with the
    # generated-boundary I2V continuation contract.
    "5b-commercial": (
        "checkpoints_5b/commercial_dense_noid_384x672_step29000.safetensors"
    ),
    "5b-full": "checkpoints_5b/v2v_id_target_704x1280_step27000.safetensors",
}
WAN_REPOS = {
    "t2v": "Wan-AI/Wan2.1-T2V-14B",
    "i2v": "Wan-AI/Wan2.1-I2V-14B-480P",
    "5b": "Wan-AI/Wan2.2-TI2V-5B",
}
WAN_REVISIONS = {
    "t2v": "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4",
    "i2v": "6b73f84e66371cdfe870c72acd6826e1d61cf279",
    "5b": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
}
VISTA4D_BUNDLE = "evaluation/vista4d_50_inputs.npz"
DEMO_BUNDLE = "evaluation/syn4d_flying_group_2chunk_demo.npz"
EXAMPLE_PREFIX = "examples/robot_davis10"
EXAMPLES_REVISION = "974e77018656957a6baf784cbfe9f9788186b5f3"


def cache_root(value=None) -> Path:
    value = value or os.environ.get("HOLO4D_CACHE_DIR")
    return Path(value).expanduser() if value else Path.home() / ".cache" / "holo4d"


def download_release_file(filename: str, *, root: Path, repo_id=None) -> Path:
    return Path(hf_hub_download(
        repo_id=repo_id or os.environ.get("HOLO4D_RELEASE_REPO", RELEASE_REPO),
        filename=filename, cache_dir=root / "huggingface",
    ))


def download_checkpoint(kind: str, *, root: Path) -> Path:
    return download_release_file(CHECKPOINT_FILES[kind], root=root)


def download_wan(kind: str, *, root: Path) -> Path:
    return Path(snapshot_download(
        repo_id=WAN_REPOS[kind],
        revision=os.environ.get(
            f"HOLO4D_WAN_{kind.upper()}_REVISION",
            WAN_REVISIONS[kind],
        ),
        cache_dir=root / "huggingface",
    ))


def prepare_examples(*, root: Path) -> Path:
    """Download ten portable examples without the repository's large weights."""
    destination = root / "examples" / "robot_davis10"
    snapshot_download(
        repo_id=os.environ.get("HOLO4D_RELEASE_REPO", RELEASE_REPO),
        revision=os.environ.get("HOLO4D_EXAMPLES_REVISION", EXAMPLES_REVISION),
        allow_patterns=[f"{EXAMPLE_PREFIX}/*"],
        local_dir=root / "example_download",
    )
    source = root / "example_download" / EXAMPLE_PREFIX
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format") != "holo4d-dense-example-v1" or len(manifest.get("examples", [])) != 10:
        raise ValueError("Incomplete or unknown example release")
    # The cache tree contains only Hub-downloaded files; preserve local user
    # modifications rather than overwrite an existing prepared example.
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    from holo4d.examples_io import resolve_member

    prepared = json.loads((destination / "manifest.json").read_text())
    if prepared.get("format") != manifest["format"] or len(prepared.get("examples", [])) != 10:
        raise ValueError("Existing local examples are incomplete; use a new HOLO4D_CACHE_DIR")
    for entry in prepared["examples"]:
        if not resolve_member(destination, entry["condition"]).is_file():
            raise ValueError("Existing local examples lack a condition file; use a new HOLO4D_CACHE_DIR")
    return destination / "manifest.json"


def _valid_vista4d_tree(path: Path) -> bool:
    try:
        manifest = json.loads((path / "bundle_manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("format") == FORMAT and all((
        (path / "vista4d_eval" / "metadata.csv").is_file(),
        (path / "frames").is_dir(), (path / "cow").is_dir(), (path / "vggt").is_dir(),
    ))


def prepare_vista4d(*, root: Path) -> Path:
    destination = root / "evaluation" / "vista4d_50_inputs"
    if _valid_vista4d_tree(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"incomplete Vista4D cache at {destination}; move it aside and retry")
    archive = download_release_file(VISTA4D_BUNDLE, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="vista4d_50_inputs.", dir=destination.parent))
    try:
        args = bundle_parser().parse_args(["extract", str(archive), "--output", str(temporary)])
        with contextlib.redirect_stdout(sys.stderr):
            extract(args)
        if not _valid_vista4d_tree(temporary):
            raise RuntimeError("extracted Vista4D bundle failed validation")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("checkpoint", "wan"):
        child = sub.add_parser(command)
        choices = CHECKPOINT_FILES if command == "checkpoint" else WAN_REPOS
        child.add_argument("kind", choices=tuple(choices))
    sub.add_parser("vista4d")
    sub.add_parser("demo")
    sub.add_parser("examples")
    child = sub.add_parser("file")
    child.add_argument("filename")
    args = parser.parse_args()
    root = cache_root(args.cache_dir)
    if args.command == "checkpoint":
        result = download_checkpoint(args.kind, root=root)
    elif args.command == "wan":
        result = download_wan(args.kind, root=root)
    elif args.command == "vista4d":
        result = prepare_vista4d(root=root)
    elif args.command == "demo":
        result = download_release_file(DEMO_BUNDLE, root=root)
    elif args.command == "examples":
        result = prepare_examples(root=root)
    else:
        result = download_release_file(args.filename, root=root)
    # Keep the snapshot filename rather than resolving the Hugging Face cache
    # symlink to its extensionless blob. The runtime uses the suffix to choose
    # the safetensors loader.
    print(result.absolute())


if __name__ == "__main__":
    main()
