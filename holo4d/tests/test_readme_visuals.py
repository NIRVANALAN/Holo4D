"""README media must be labelled and must not eagerly fetch every video."""
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

from scripts.build_readme_visuals import gallery_document


class VideoTags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = []

    def handle_starttag(self, tag, attrs):
        if tag == "video":
            self.videos.append(dict(attrs))


def example():
    return dict(id="example", dataset="DAVIS", backend="4rc", height=384,
                width=672, trajectory="pan_left", caption="Bear <moving> & walking",
                prediction_note="Archived qualitative reference",
                source_poster="source.jpg", prediction_poster="prediction.jpg",
                source_url="https://example.test/source.mp4",
                prediction_url="https://example.test/prediction.mp4")


def test_gallery_uses_posters_and_click_only_media():
    text = gallery_document([example()])
    parser = VideoTags()
    parser.feed(text)
    assert len(parser.videos) == 2
    for attrs in parser.videos:
        assert attrs["preload"] == "none"
        assert "src" not in attrs
        assert attrs["poster"].endswith(".jpg")
        assert attrs["data-src"].endswith(".mp4")
    assert "v.removeAttribute('src')" in text


def test_gallery_labels_archived_outputs_and_escapes_captions():
    text = gallery_document([example()])
    assert "Archived qualitative reference" in text
    assert "Bear &lt;moving&gt; &amp; walking" in text
    assert "EXAMPLE=example bash inference/demo_examples.sh" in text


def test_local_gallery_videos_are_publishable_release_assets():
    root = Path(__file__).resolve().parents[2]
    gallery_root = root / "media/docs"
    manifest = json.loads((gallery_root / "manifest.json").read_text())
    local_videos = []
    for entry in manifest["examples"]:
        url = entry["prediction_url"]
        if url.startswith(("http://", "https://")):
            continue
        path = gallery_root / url
        assert path.is_file(), f"Missing gallery video: {path}"
        local_videos.append(str(path.relative_to(root)))
    assert local_videos, "Expected fresh, bundled validation previews"
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *local_videos],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout or result.stderr
