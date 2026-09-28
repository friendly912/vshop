import json
import os
import shutil

import pytest

from vton import evaluate


def test_paired_identical_outputs_score_perfect(dataset, tmp_path):
    gt = dataset / "test" / "image"
    pred = tmp_path / "pred"
    shutil.copytree(gt, pred)
    out = tmp_path / "result.json"
    evaluate.main(["--pred", str(pred), "--gt", str(gt), "--mode", "paired",
                   "--size", "64", "48", "--no-lpips", "--out", str(out)])
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["images"] == 2 and result["unmatched_outputs"] == 0
    assert result["ssim"] == pytest.approx(1.0, abs=1e-4)


def test_paired_without_matches_fails(dataset, tmp_path):
    (tmp_path / "pred").mkdir()
    with pytest.raises(SystemExit):
        evaluate.main(["--pred", str(tmp_path / "pred"), "--gt", str(dataset / "test" / "image"),
                       "--mode", "paired", "--no-lpips"])


@pytest.mark.skipif(os.environ.get("VTON_NETWORK_TESTS") != "1",
                    reason="downloads LPIPS/Inception weights; set VTON_NETWORK_TESTS=1")
def test_pretrained_metrics_run(dataset, tmp_path):
    gt = dataset / "test" / "image"
    out = tmp_path / "r.json"
    evaluate.main(["--pred", str(gt), "--gt", str(gt), "--mode", "paired", "--size", "64", "48",
                   "--out", str(out)])
    assert json.loads(out.read_text())["lpips"] == pytest.approx(0.0, abs=1e-4)
    evaluate.main(["--pred", str(gt), "--gt", str(gt), "--mode", "unpaired", "--size", "64", "48",
                   "--out", str(out)])
    assert "fid" in json.loads(out.read_text())
