"""Drop-in replacements for ops that can't run on the phone (or on CPU at all).

- grid_sample_gather: F.grid_sample (bilinear) as a 1-D GATHER on the flattened image. LiteRT
  has no grid_sample op; this formulation passed the GPU-delegate analyzer in ml/warp_spike
  and matched torch exactly.
- correlation: the 7x7 cost volume from PWC-Net style flow networks. The original DM-VTON
  version is a CUDA-only cupy kernel; this uses only pad/slice/mul/mean.
"""

import torch
import torch.nn.functional as F


def grid_sample_gather(inp: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear",
                       padding_mode: str = "zeros", align_corners: bool = False) -> torch.Tensor:
    """Same result as F.grid_sample(inp, grid, 'bilinear', padding_mode, align_corners) for
    padding_mode in {'zeros', 'border'}. inp [N,C,H,W], grid [N,Hg,Wg,2] -> [N,C,Hg,Wg]."""
    if mode != "bilinear" or padding_mode not in ("zeros", "border"):
        raise NotImplementedError(f"mode={mode}, padding_mode={padding_mode}")
    n, c, h, w = inp.shape
    hg, wg = grid.shape[1], grid.shape[2]
    gx, gy = grid[..., 0], grid[..., 1]
    if align_corners:
        x = (gx + 1) * ((w - 1) * 0.5)
        y = (gy + 1) * ((h - 1) * 0.5)
    else:
        x = ((gx + 1) * w - 1) * 0.5
        y = ((gy + 1) * h - 1) * 0.5
    if padding_mode == "border":
        x = x.clamp(0, w - 1)
        y = y.clamp(0, h - 1)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx = x - x0
    fy = y - y0
    corners = ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
               (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy))

    outs = []
    for b in range(n):  # export uses batch 1; the loop keeps training-time use correct
        flat = inp[b].reshape(c, h * w).transpose(0, 1)  # [H*W, C]
        acc = None
        for xi, yi, wt in corners:
            xb, yb = xi[b], yi[b]
            valid = ((xb >= 0) & (xb <= w - 1) & (yb >= 0) & (yb <= h - 1)).to(inp.dtype)
            idx = (yb.clamp(0, h - 1).to(torch.int32) * w + xb.clamp(0, w - 1).to(torch.int32))
            px = torch.index_select(flat, 0, idx.reshape(-1)).reshape(hg, wg, c)
            term = px * (wt[b] * valid)[..., None]
            acc = term if acc is None else acc + term
        outs.append(acc.permute(2, 0, 1))
    return torch.stack(outs)


def correlation(first: torch.Tensor, second: torch.Tensor, intStride: int = 1,
                max_disp: int = 3) -> torch.Tensor:
    """Cost volume: channel (dy+3)*7 + (dx+3) = mean_c first[y,x] * second[y+dy, x+dx],
    zero padded. Matches DM-VTON's FunctionCorrelation (stride 1)."""
    if intStride != 1:
        raise NotImplementedError("only stride 1 is used by DM-VTON")
    d = max_disp
    h, w = first.shape[2], first.shape[3]
    padded = F.pad(second, (d, d, d, d))
    vols = []
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            shifted = padded[:, :, d + dy:d + dy + h, d + dx:d + dx + w]
            vols.append((first * shifted).mean(1, keepdim=True))
    return torch.cat(vols, 1)
