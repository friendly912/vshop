# ml/

| Path | Purpose |
|---|---|
| `vton/data.py` | `TryOnDataset` for the VITON-HD folder layout, plus a layout checker |
| `vton/preprocess.py` | Pose, parsing and garment masks for data without annotations |
| `vton/metrics.py` | SSIM, LPIPS, FID and KID |
| `vton/evaluate.py` | Score a folder of try-on outputs |
| `tests/` | pytest suite on synthetic data; CI runs it (`.github/workflows/ml.yml`) |
| `export_placeholder_model.py`, `export_bench_model.py` | Models bundled into the Android app |
| `warp_spike/` | Warp formulation test (see its README) |

Setup: `pip install -r ml/requirements-train.txt`. On Linux, MediaPipe also needs
`sudo apt-get install libegl1 libgles2`, even on machines without a display. Run everything below from `ml/` so `vton`
is importable, or set `PYTHONPATH=ml`.

## Dataset layout

Every source (VITON-HD, DressCode, your own photo shoot) is converted to this layout:

```
<root>/test_pairs.txt            "person.jpg cloth.jpg" per line (unpaired evaluation)
<root>/test/image/               person photos
<root>/test/cloth/               garment product photos, same file name as the wearer
<root>/test/cloth-mask/          optional
<root>/test/parse-mp/            optional, generated
<root>/test/pose-mp/             optional, generated
```

```
python -m vton.data --root <root> --split test                 # check the layout
python -m vton.preprocess --root <root> --split test           # fill in missing annotations
```

Parsing uses MediaPipe's 6-class segmenter (the same one as the app), not the 20-class LIP
parsing VITON-HD ships. That's enough for parser-free training (phase 2), where parsing is only
used to build training inputs. Garment masks assume a white background.

## Evaluation

```
python -m vton.evaluate --pred <outputs> --gt <root>/test/image --mode paired     # SSIM, LPIPS
python -m vton.evaluate --pred <outputs> --gt <root>/test/image --mode unpaired   # FID, KID
```

Papers report these at 1024x768 on VITON-HD test (2,032 images). Use the same resolution and
image count for any comparison. FID on a few hundred images is only good for comparing your
own runs with each other.

## Licensing reminder

VITON-HD and DressCode are non-commercial. Whether this code may be used on them for a
commercial product is a phase 0 decision; the code itself doesn't bundle any dataset.
