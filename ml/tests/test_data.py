import pytest
import torch
from torch.utils.data import DataLoader

from vton.data import TryOnDataset, validate_layout


def test_layout_report_ok(dataset):
    r = validate_layout(dataset, "test")
    assert r.ok, r.missing
    assert r.pairs == 2
    assert r.optional == {"cloth-mask": True, "parse-mp": True, "pose-mp": True}


def test_paired_items(dataset):
    ds = TryOnDataset(dataset, "test", size=(32, 24))
    assert len(ds) == 2
    item = ds[0]
    assert item["person_name"] == item["cloth_name"] == "a.jpg"
    assert item["person"].shape == item["cloth"].shape == (3, 32, 24)
    assert item["person"].min() >= -1 and item["person"].max() <= 1
    assert item["cloth_mask"].shape == (1, 32, 24)
    assert set(item["cloth_mask"].unique().tolist()) <= {0.0, 1.0}
    assert item["parse"].dtype == torch.int64 and item["parse"].max() <= 5
    assert item["pose"].shape == (33, 4) and item["pose"][0, 3] == pytest.approx(0.9)
    assert ds[1]["pose"].abs().sum() == 0  # no person detected -> zeros


def test_unpaired_uses_pairs_file(dataset):
    ds = TryOnDataset(dataset, "test", size=(32, 24), paired=False)
    assert [(ds[i]["person_name"], ds[i]["cloth_name"]) for i in range(2)] == [("a.jpg", "b.jpg"), ("b.jpg", "a.jpg")]


def test_dataloader_batches(dataset):
    batch = next(iter(DataLoader(TryOnDataset(dataset, "test", size=(32, 24)), batch_size=2)))
    assert batch["person"].shape == (2, 3, 32, 24)
    assert batch["pose"].shape == (2, 33, 4)
    assert batch["person_name"] == ["a.jpg", "b.jpg"]


def test_missing_files_are_reported(dataset):
    (dataset / "test" / "cloth-mask" / "b.png").unlink()
    r = validate_layout(dataset, "test")
    assert not r.ok and r.missing == ["cloth-mask/b.jpg"]
    with pytest.raises(FileNotFoundError, match="cloth-mask/b.jpg"):
        TryOnDataset(dataset, "test")


def test_missing_required_dir(tmp_path):
    r = validate_layout(tmp_path, "test")
    assert not r.ok and len(r.missing) == 2
