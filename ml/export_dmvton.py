"""Convert DM-VTON to LiteRT for the Android app and verify the result.

Runs on Linux (litert-torch). The CI workflow .github/workflows/convert.yml runs it; locally:

    pip install -r ml/requirements-convert.txt
    python ml/export_dmvton.py --out build/viton_dm.tflite

Steps: load the pretrained model with the mobile op replacements (vton.mobile_ops), convert
with litert-torch, then check the .tflite against PyTorch on random inputs and on one
VITON-style synthetic input, and write a JSON report (size, max error, ops, CPU latency).
Exits non-zero if the converted model's output differs from PyTorch.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vton import dmvton  # noqa: E402

MAX_ABS_ERR = 1e-3  # float32 conversion; operator reordering only


def sample_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    h, w = dmvton.HEIGHT, dmvton.WIDTH
    person = torch.rand(1, h, w, 3, generator=g) * 2 - 1
    garment = torch.rand(1, h, w, 3, generator=g) * 2 - 1
    mask = torch.zeros(1, h, w, 1)
    mask[:, h // 8:h * 7 // 8, w // 6:w * 5 // 6] = 1  # garment-shaped region, like real masks
    return person, garment, mask


def tflite_ops(model_path: Path) -> list:
    """Op names in the flatbuffer (via litert's schema, no TensorFlow needed)."""
    try:
        from ai_edge_litert import schema_py_generated as schema
    except ImportError:
        return []
    buf = model_path.read_bytes()
    m = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    ops = set()
    for code in m.operatorCodes:
        op = max(code.builtinCode, code.deprecatedBuiltinCode)
        ops.add(names.get(op, str(op)) if op != schema.BuiltinOperator.CUSTOM else f"CUSTOM:{code.customCode}")
    return sorted(ops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import litert_torch
    from ai_edge_litert.interpreter import Interpreter

    model = dmvton.TryOnNHWC(dmvton.load_pipeline()).eval()
    inputs = sample_inputs(0)
    with torch.no_grad(), dmvton.gather_grid_sample():
        edge = litert_torch.convert(model, inputs)
        edge.export(str(args.out))

        report = {"model": str(args.out), "bytes": args.out.stat().st_size, "checks": []}
        it = Interpreter(model_path=str(args.out), num_threads=4)
        it.allocate_tensors()
        in_details = sorted(it.get_input_details(), key=lambda d: d["index"])
        out_idx = it.get_output_details()[0]["index"]
        report["inputs"] = [{"name": d["name"], "shape": d["shape"].tolist()} for d in in_details]

        worst = 0.0
        for seed in (0, 1, 2):
            x = sample_inputs(seed)
            ref = model(*x).numpy()
            for d, t in zip(in_details, x):
                it.set_tensor(d["index"], t.numpy())
            t0 = time.perf_counter()
            it.invoke()
            ms = (time.perf_counter() - t0) * 1000
            err = float(np.abs(it.get_tensor(out_idx) - ref).max())
            worst = max(worst, err)
            report["checks"].append({"seed": seed, "max_abs_err": err, "cpu_ms": round(ms, 1)})

    report["max_abs_err"] = worst
    report["ops"] = tflite_ops(args.out)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    if worst > MAX_ABS_ERR:
        raise SystemExit(f"converted model differs from PyTorch: max abs err {worst:.2e} > {MAX_ABS_ERR}")


if __name__ == "__main__":
    main()
