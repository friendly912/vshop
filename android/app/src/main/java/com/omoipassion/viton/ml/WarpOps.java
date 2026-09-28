package com.omoipassion.viton.ml;

/**
 * Bilinear warp outside the model ("variant D" in ml/warp_spike): flow model on GPU →
 * this warp on CPU → generator on GPU. Matches torch grid_sample with mode=bilinear,
 * padding_mode=zeros, align_corners=False. All arrays are NHWC with batch 1.
 */
public final class WarpOps {

    private WarpOps() {
    }

    /**
     * @param img  [h*w*c] source image
     * @param grid [h*w*2] sampling grid (x, y) normalized to [-1, 1]
     * @param out  [h*w*c] destination
     */
    public static void bilinear(float[] img, float[] grid, int h, int w, int c, float[] out) {
        for (int p = 0, n = h * w; p < n; p++) {
            float x = ((grid[2 * p] + 1f) * w - 1f) * 0.5f;
            float y = ((grid[2 * p + 1] + 1f) * h - 1f) * 0.5f;
            int x0 = (int) Math.floor(x);
            int y0 = (int) Math.floor(y);
            float fx = x - x0;
            float fy = y - y0;
            int o = p * c;
            for (int k = 0; k < c; k++) {
                out[o + k] = 0f;
            }
            accumulate(img, h, w, c, x0, y0, (1 - fx) * (1 - fy), out, o);
            accumulate(img, h, w, c, x0 + 1, y0, fx * (1 - fy), out, o);
            accumulate(img, h, w, c, x0, y0 + 1, (1 - fx) * fy, out, o);
            accumulate(img, h, w, c, x0 + 1, y0 + 1, fx * fy, out, o);
        }
    }

    private static void accumulate(float[] img, int h, int w, int c, int x, int y, float wt,
                                   float[] out, int o) {
        if (x < 0 || y < 0 || x >= w || y >= h || wt == 0f) {
            return; // zeros padding
        }
        int s = (y * w + x) * c;
        for (int k = 0; k < c; k++) {
            out[o + k] += wt * img[s + k];
        }
    }
}
