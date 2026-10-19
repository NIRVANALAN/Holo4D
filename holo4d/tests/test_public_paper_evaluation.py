"""CPU gates for public paper protocols; no datasets, weights or GPU needed."""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from holo4d.tools.eval_vista4d_camera_pi3 import apply_similarity, pose_stabilized_similarity
from holo4d.tools.prepare_davis_vbench_inputs import read_captions, read_pairs
from holo4d.tools.summarize_davis_vbench import DIMENSIONS, summarize
from holo4d.tools.verify_metric_replay import compare

ROOT = Path(__file__).resolve().parents[2]


def test_shell_wrapper_paths_and_syntax():
    for name in ("vggt_omega", "vipe", "da3"):
        path = ROOT / f"eval/common/eval_camera_metrics_{name}.sh"
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert f'${{SCRIPT_DIR}}/{name}_camera_eval.py' in path.read_text()
        assert (ROOT / f"eval/common/{name}_camera_eval.py").is_file()
    text = (ROOT / "eval/common/eval_camera_metrics_vggt_omega.sh").read_text()
    assert 'TARGET_W="${TARGET_W:-672}"' in text
    assert 'TARGET_H="${TARGET_H:-384}"' in text


def test_pi3_source_fit_transfers_one_sim3_to_unseen_target():
    source = np.tile(np.eye(4), (49, 1, 1))
    source[:, 0, 3] = np.linspace(0, 1, 49)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([1., 2., 3.])
    reference = apply_similarity(source, 2., rotation, translation)
    scale, fitted_rotation, fitted_translation = pose_stabilized_similarity(source, reference)
    np.testing.assert_allclose(scale, 2.)
    np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-12)
    target = source.copy(); target[:, 2, 3] += 4.
    np.testing.assert_allclose(apply_similarity(target, scale, fitted_rotation, fitted_translation),
                               apply_similarity(target, 2., rotation, translation), atol=1e-12)


def test_pi3_degenerate_source_is_not_silently_target_fitted():
    poses = np.tile(np.eye(4), (49, 1, 1))
    with pytest.raises(RuntimeError, match="scale is not identifiable"):
        pose_stabilized_similarity(poses, poses)


def test_davis_requires_full_unique_pairs_and_nonempty_captions(tmp_path):
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "trajectory", "caption"])
        writer.writeheader()
        writer.writerows(dict(video=f"scene{i//2}", trajectory=str(i%2), caption="Robot moving") for i in range(100))
    assert len(read_pairs(path)) == len(read_captions(path)) == 100
    path.write_text(path.read_text().replace("Robot moving", "", 1))
    with pytest.raises(ValueError, match="nonempty"):
        read_captions(path)


def test_vbench_cohort_completeness_and_duplicate_detection(tmp_path):
    for dimension in DIMENSIONS:
        (tmp_path / f"{dimension}_eval_results.json").write_text(json.dumps({
            dimension: [.5, [dict(video_path=f"case{i}.mp4", video_results=.5) for i in range(100)]]}))
    result = summarize(tmp_path, 100)
    assert all(row["n"] == 100 for row in result.values())
    path = tmp_path / f"{DIMENSIONS[0]}_eval_results.json"
    payload = json.loads(path.read_text()); payload[DIMENSIONS[0]][1][1]["video_path"] = "case0.mp4"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(tmp_path, 100)


def test_replay_comparison_rejects_wrong_counts_and_nonfinite():
    compare(dict(mean=.01657, n=54), dict(mean=.01657, n=54))
    with pytest.raises(ValueError):
        compare(dict(mean=.01657, n=54), dict(mean=.01657, n=100))
    with pytest.raises(ValueError, match="non-finite"):
        compare(float("nan"), float("nan"))
