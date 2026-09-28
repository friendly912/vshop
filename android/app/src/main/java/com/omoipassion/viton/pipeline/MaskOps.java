package com.omoipassion.viton.pipeline;

import java.util.List;

/**
 * Pure-Java mask math for the preserve-composite step (no Android types, unit-tested).
 * Masks are row-major float[w*h] in [0, 1]; pixels are ARGB int[].
 */
final class MaskOps {

    /** A hand region in mask pixel coordinates. */
    static final class Circle {
        final float cx;
        final float cy;
        final float r;

        Circle(float cx, float cy, float r) {
            this.cx = cx;
            this.cy = cy;
            this.r = r;
        }
    }

    private MaskOps() {
    }

    /**
     * Where to keep the original photo (1) instead of the try-on output (0):
     * hair, face, accessories, hands (body skin inside a hand circle), and background
     * farther than {@code bgMargin} px from the person.
     */
    static float[] keepMask(float[] background, float[] hair, float[] bodySkin, float[] faceSkin,
                            float[] others, int w, int h, List<Circle> hands, int bgMargin) {
        float[] person = new float[w * h];
        for (int i = 0; i < person.length; i++) {
            person[i] = 1f - background[i];
        }
        float[] near = dilate(person, w, h, bgMargin);

        float[] keep = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                float k = Math.max(hair[i], Math.max(faceSkin[i], others[i]));
                if (insideAny(hands, x + 0.5f, y + 0.5f)) {
                    k = Math.max(k, bodySkin[i]);
                }
                k = Math.max(k, 1f - near[i]);
                keep[i] = k;
            }
        }
        return keep;
    }

    private static boolean insideAny(List<Circle> circles, float x, float y) {
        for (Circle c : circles) {
            float dx = x - c.cx;
            float dy = y - c.cy;
            if (dx * dx + dy * dy <= c.r * c.r) {
                return true;
            }
        }
        return false;
    }

    /** Separable max filter with a (2r+1)^2 square window. */
    static float[] dilate(float[] m, int w, int h, int r) {
        if (r <= 0) {
            return m.clone();
        }
        float[] tmp = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float v = 0f;
                for (int k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++) {
                    v = Math.max(v, m[y * w + k]);
                }
                tmp[y * w + x] = v;
            }
        }
        float[] out = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float v = 0f;
                for (int k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k++) {
                    v = Math.max(v, tmp[k * w + x]);
                }
                out[y * w + x] = v;
            }
        }
        return out;
    }

    /** Separable box blur (edge-clamped), used to feather seams. */
    static float[] boxBlur(float[] m, int w, int h, int r) {
        if (r <= 0) {
            return m.clone();
        }
        float norm = 1f / (2 * r + 1);
        float[] tmp = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float s = 0f;
                for (int k = -r; k <= r; k++) {
                    s += m[y * w + clamp(x + k, w)];
                }
                tmp[y * w + x] = s * norm;
            }
        }
        float[] out = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float s = 0f;
                for (int k = -r; k <= r; k++) {
                    s += tmp[clamp(y + k, h) * w + x];
                }
                out[y * w + x] = s * norm;
            }
        }
        return out;
    }

    /**
     * Blends in place: {@code tryOn = original * a + tryOn * (1 - a)} over a crop at
     * ({@code left}, {@code top}) of a {@code fullW x fullH} image, where the keep mask
     * covers the full image at {@code mw x mh} and is sampled bilinearly.
     */
    static void blendInto(int[] tryOn, int[] original, int cw, int ch, int left, int top,
                          int fullW, int fullH, float[] mask, int mw, int mh) {
        float sx = (float) mw / fullW;
        float sy = (float) mh / fullH;
        for (int y = 0; y < ch; y++) {
            float v = (top + y + 0.5f) * sy - 0.5f;
            for (int x = 0; x < cw; x++) {
                float u = (left + x + 0.5f) * sx - 0.5f;
                float a = sample(mask, mw, mh, u, v);
                int i = y * cw + x;
                if (a <= 0f) {
                    continue;
                }
                if (a >= 1f) {
                    tryOn[i] = original[i];
                    continue;
                }
                tryOn[i] = mix(original[i], tryOn[i], a);
            }
        }
    }

    static float sample(float[] m, int w, int h, float u, float v) {
        u = Math.max(0f, Math.min(u, w - 1f));
        v = Math.max(0f, Math.min(v, h - 1f));
        int x0 = (int) u;
        int y0 = (int) v;
        int x1 = Math.min(x0 + 1, w - 1);
        int y1 = Math.min(y0 + 1, h - 1);
        float fx = u - x0;
        float fy = v - y0;
        float top = m[y0 * w + x0] * (1 - fx) + m[y0 * w + x1] * fx;
        float bottom = m[y1 * w + x0] * (1 - fx) + m[y1 * w + x1] * fx;
        return top * (1 - fy) + bottom * fy;
    }

    private static int mix(int p, int q, float a) {
        float b = 1f - a;
        int r = Math.round(((p >> 16) & 0xFF) * a + ((q >> 16) & 0xFF) * b);
        int g = Math.round(((p >> 8) & 0xFF) * a + ((q >> 8) & 0xFF) * b);
        int bl = Math.round((p & 0xFF) * a + (q & 0xFF) * b);
        return 0xFF000000 | (r << 16) | (g << 8) | bl;
    }

    private static int clamp(int i, int n) {
        return i < 0 ? 0 : (i >= n ? n - 1 : i);
    }
}
