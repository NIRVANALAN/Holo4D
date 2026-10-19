"""Build the README GIFs from the project-page videos on the Hub.

results.gif: 2x2 grid of the project-page "interleaved" players (source + 3D tracks, point-tracking condition, synthesized view;
clean and track-overlay loops alternate). comparison.gif: Source / Vista4D / Holo4D.

CPU only (FFmpeg + Pillow). media/docs/teaser.jpg and overview.jpg are exports of the paper's Figures 1 and 2.
Usage: python scripts/build_readme_media.py [--output-dir media/docs] [--work-dir /tmp/holo4d_readme]
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ("https://huggingface.co/yslan/Holo4D-2026/resolve/9778c4cc4adbb7d5c50fa5edae57f8bca2581bf1/"
         "project_page/media_20261012/comparison/")
HERO = ["filtered20261011_v2/native110/friends-restaurant__side-dolly", "filtered20261011_v2/native110/golf__dolly-out",
        "filtered20261011_v2/davis/bear__pan_left",
        "filtered20261011_v2/robot/004_TRI__failure__2023-09-21__Thu_Sep_21_10-05-35_2023__static_2__25947356__safe_near_orbit"]
SCENES = {"golf": "golf/dolly-out", "restaurant": "friends-restaurant/side-dolly",
          "desert": "drive-desert/back-follow", "fashion": "fashion-walk/arc-left-right"}


def font(size: int):
    for name in ("Arial Bold.ttf", "DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fetch(work: Path, scene: str, kind: str) -> Path:
    path = work / f"{scene}_{kind}.mp4"
    if not path.is_file():
        urllib.request.urlretrieve(f"{MEDIA}{SCENES[scene]}/{kind}.mp4", path)
    return path


def frames(mp4: Path, width: int, count: int = 49) -> list[Image.Image]:
    out = Path(str(mp4) + "_frames")
    shutil.rmtree(out, ignore_errors=True); out.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-vf", f"scale={width}:-2",
                    "-frames:v", str(count), str(out / "%03d.png")], check=True)
    return [Image.open(f).convert("RGB") for f in sorted(glob.glob(str(out / "*.png")))]


def label(image: Image.Image, text: str, ours: bool) -> Image.Image:
    image = image.copy(); draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([0, 0, image.width, 19], fill=(0, 0, 0, 140))
    draw.text((6, 3), text, font=font(13), fill=(143, 195, 255) if ours else (255, 255, 255))
    return image


def grid(work: Path, rows, width: int, out: Path, gap: int = 3) -> None:
    """rows: [[(scene, kind, label, ours), ...], ...]; every other frame of 49, played at 8 FPS."""
    tiles = [[(frames(fetch(work, s, k), width), t, o) for s, k, t, o in row] for row in rows]
    n = min(len(f) for row in tiles for f, _, _ in row); tw, th = tiles[0][0][0][0].size
    canvas_w = len(tiles[0]) * tw + (len(tiles[0]) - 1) * gap; canvas_h = len(tiles) * th + (len(tiles) - 1) * gap
    seq = work / (out.stem + "_frames"); shutil.rmtree(seq, ignore_errors=True); seq.mkdir()
    for k in range(0, n, 2):
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        for i, row in enumerate(tiles):
            for j, (fr, text, ours) in enumerate(row):
                canvas.paste(label(fr[k], text, ours), (j * (tw + gap), i * (th + gap)))
        canvas.save(seq / f"{k // 2:03d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "8", "-i", str(seq / "%03d.png"),
                    "-filter_complex", "split[x][y];[x]palettegen=max_colors=128[p];[y][p]paletteuse=dither=bayer:bayer_scale=5",
                    str(out)], check=True)
    print(f"[readme] {out} {os.path.getsize(out) // 1024} KB {canvas_w}x{canvas_h}")


def hero(work: Path, out: Path, width: int = 380, step: int = 3, fps: int = 6, colors: int = 96) -> None:
    """Every `step`-th frame of each 98-frame interleaved player (clean loop then overlay loop), 2x2 grid."""
    base = MEDIA.replace("comparison/", "")
    inputs = []
    for path in HERO:
        local = work / (path.rsplit("/", 1)[-1] + "_interleaved.mp4")
        if not local.is_file():
            urllib.request.urlretrieve(f"{base}{path}/interleaved.mp4", local)
        inputs += ["-i", str(local)]
    tile = lambda i: f"[{i}]scale={width}:-2:flags=lanczos,select='not(mod(n\\,{step}))',setpts=N/{fps}/TB[v{i}]"
    graph = ";".join(tile(i) for i in range(4)) + (f";[v0][v1]hstack[r1];[v2][v3]hstack[r2];[r1][r2]vstack,fps={fps},split[x][y];"
             f"[x]palettegen=max_colors={colors}:stats_mode=diff[p];[y][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, str(out)], check=True)
    print(f"[readme] {out} {os.path.getsize(out) // 1024} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "media/docs")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/holo4d_readme"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    hero(args.work_dir, args.output_dir / "results.gif")
    trio = lambda s: [(s, "source", "Source", False), (s, "baseline", "Vista4D", False), (s, "clean", "Holo4D (ours)", True)]
    grid(args.work_dir, [trio("golf"), trio("desert")], 280, args.output_dir / "comparison.gif")


if __name__ == "__main__":
    main()
