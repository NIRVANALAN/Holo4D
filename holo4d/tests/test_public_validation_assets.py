from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "assets/vista4d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_public_vista4d_assets_match_manifest() -> None:
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    outputs = manifest["outputs"]
    assert len(outputs) == 10
    assert len({row["source"] for row in outputs}) == 5
    assert {row["frames"] for row in outputs} == {49}
    assert manifest["checkpoint"] == "t2v_49_chunk.safetensors"

    for row in outputs:
        path = ASSET_ROOT / row["file"]
        assert path.is_file()
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]
        reader = imageio.get_reader(path)
        try:
            frame = reader.get_data(0)
            assert frame.shape[:2] == (384, 672)
            assert reader.count_frames() == 49
        finally:
            reader.close()
