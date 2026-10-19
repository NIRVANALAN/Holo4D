"""GitHub README previews must be animated images, not embedded video tags."""
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("example", ["tri_static_high", "figure0_orbit"])
def test_animated_rollout_preview(example):
    relative = f"assets/demo/robot_long/{example}.gif"
    preview = ROOT / relative
    with Image.open(preview) as image:
        assert image.format == "GIF"
        assert image.is_animated
        assert image.n_frames > 20
        assert image.width == 400
        assert image.info["loop"] == 0
    assert preview.stat().st_size < 3_000_000
    readme = (ROOT / "README.md").read_text()
    assert f"]({relative})" in readme
    assert f"](assets/demo/robot_long/{example}.mp4)" in readme
