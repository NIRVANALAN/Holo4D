"""Build the local example gallery (media/docs/gallery.html, posters, manifest) from existing videos.
README GIFs/teaser come from scripts/build_readme_media.py.

CPU only: no model loading, geometry generation, or diffusion sampling.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REVISION = "974e77018656957a6baf784cbfe9f9788186b5f3"
HUB_BASE = f"https://huggingface.co/yslan/Holo4D-2026/resolve/{REVISION}/examples/robot_davis10"


def frame_at(path: Path, index: int) -> Image.Image:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
         "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-threads", "1",
         "-f", "image2pipe", "-c:v", "png", "-"],
        capture_output=True, check=True,
    )
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def preview_frames(path: Path, width: int, height: int) -> list[Image.Image]:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
         "-vf", f"fps=8,scale={width}:{height}", "-threads", "1",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        check=True, capture_output=True,
    )
    size = width * height * 3
    return [Image.frombytes("RGB", (width, height), result.stdout[start:start + size])
            for start in range(0, len(result.stdout), size)]


def export_preview(source: Path, destination: Path) -> None:
    # Presentation copy only. Do not overwrite the inference artifact.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-threads", "1", "-i", str(source),
         "-an", "-frames:v", "49", "-vf", "scale=w='min(672,iw)':h=-2",
         "-r", "16", "-c:v", "libx264", "-threads", "1", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)],
        check=True,
    )


def gallery_document(entries: list[dict]) -> str:
    cards = []
    for entry in entries:
        identifier = html.escape(entry["id"])
        metadata = html.escape(f'{entry["dataset"]} · {entry["backend"]} · {entry["height"]}×{entry["width"]} H×W · {entry["trajectory"]}')
        command = html.escape(f'EXAMPLE={entry["id"]} bash inference/demo_examples.sh')
        cards.append(f'''<article><h2>{identifier}</h2><p>{metadata}</p>
<p class="badge">{html.escape(entry["prediction_note"])}</p>
<div class="pair"><div><h3>Source</h3><video controls playsinline preload="none" poster="{entry["source_poster"]}" data-src="{entry["source_url"]}"></video></div>
<div><h3>Novel view</h3><video controls playsinline preload="none" poster="{entry["prediction_poster"]}" data-src="{entry["prediction_url"]}"></video></div></div>
<button class="play">Load and play this pair</button><button class="copy" data-command="{command}">Copy inference command</button>
<details><summary>Caption and reproduction</summary><p>{html.escape(entry["caption"])}</p><code>{command}</code></details></article>''')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Holo4D · prepared examples</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#12243a;font:16px system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:32px 20px}h1{font-size:38px;margin-bottom:8px}p{line-height:1.6;color:#526278}article{background:white;border:1px solid #dce4ef;border-radius:16px;padding:24px;margin:24px 0}h2{font-size:22px;margin:0}h3{font-size:14px;color:#526278}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair video{width:100%;height:320px;object-fit:contain;background:#edf1f6;border-radius:10px}button{cursor:pointer;padding:11px 16px;margin:16px 8px 8px 0;border-radius:8px;border:1px solid #326ad0;color:#326ad0;background:white}.play{background:#326ad0;color:white}.badge{font-size:13px;background:#edf3ff;padding:8px;border-radius:6px}code{white-space:pre-wrap}details{margin-top:10px}a{color:#326ad0}@media(max-width:650px){.pair{grid-template-columns:1fr}.pair video{height:260px}main{padding:20px 12px}}
</style></head><body><main><h1>Holo4D examples</h1>
<p>Five DAVIS sources and five robot sources. Dense geometry, captions and target cameras are included in the prepared pack. Videos are loaded only when you select a pair; all other cards show their first frame.</p>
<p>Bear and Figure0 use fresh public 14B 50-step validation outputs. Other cards show the pack's archived qualitative references, not newly reproduced outputs or ground truth. Portrait images keep their native aspect ratio. Source media retain their upstream rights.</p>
<p><a href="../../README.md">README</a> · <a href="../../doc/visualization.md">Visualization guide</a></p>
''' + "\n".join(cards) + '''</main><script>
const cards = [...document.querySelectorAll('article')];
for (const card of cards) {
  card.querySelector('.play').addEventListener('click', () => {
    for (const other of cards) if (other !== card) for (const v of other.querySelectorAll('video')) {
      if (v.hasAttribute('src')) { v.pause(); v.removeAttribute('src'); v.load(); }
    }
    for (const v of card.querySelectorAll('video')) {
      if (!v.hasAttribute('src')) { v.src = v.dataset.src; v.load(); }
      v.currentTime = 0; v.play().catch(() => {});
    }
  });
  card.querySelector('.copy').addEventListener('click', async event => {
    const button = event.currentTarget;
    try { await navigator.clipboard.writeText(button.dataset.command); button.textContent = 'Copied'; }
    catch { window.prompt('Copy this command:', button.dataset.command); }
  });
}
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "media/docs")
    args = parser.parse_args()
    manifest = json.loads((args.bundle_root / "manifest.json").read_text())
    output = args.output_dir
    (output / "posters").mkdir(parents=True, exist_ok=True)
    (output / "generated").mkdir(exist_ok=True)
    entries = []
    for item in manifest["examples"]:
        entry = dict(item)
        example = entry["id"]
        source = args.bundle_root / entry["source_video"]
        fresh = args.validation_root / "14b" / example / "pred_rgb.mp4"
        prediction = fresh if fresh.is_file() else args.bundle_root / entry["reference_video"]
        for kind, video in (("source", source), ("prediction", prediction)):
            poster = frame_at(video, 0)
            poster.thumbnail((672, 672), Image.Resampling.LANCZOS)
            relative = f"posters/{example}_{kind}.jpg"
            poster.save(output / relative, quality=90)
            entry[f"{kind}_poster"] = relative
        entry["source_url"] = f"{HUB_BASE}/{entry['source_video']}"
        entry["prediction_note"] = "Fresh public 14B · 50 steps · CFG 5 · shift 5 · seed 42" if fresh.is_file() else "Archived qualitative reference · recipe may differ; regenerate with the command below"
        if fresh.is_file():
            destination = output / "generated" / f"{example}.mp4"
            export_preview(fresh, destination)
            entry["prediction_url"] = f"generated/{example}.mp4"
        else:
            entry["prediction_url"] = f"{HUB_BASE}/{entry['reference_video']}"
        entries.append(entry)
    (output / "gallery.html").write_text(gallery_document(entries))
    (output / "manifest.json").write_text(json.dumps({"hub_revision": REVISION, "purpose": "README presentation, not benchmark evaluation", "examples": entries}, indent=2) + "\n")
    print(f"[readme] {len(entries)} poster pairs; lazy gallery: {output}", flush=True)


if __name__ == "__main__":
    main()
