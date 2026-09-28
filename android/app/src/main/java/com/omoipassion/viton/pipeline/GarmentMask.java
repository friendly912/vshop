package com.omoipassion.viton.pipeline;

/**
 * Garment mask for a product photo on a white background: the "cloth edge" input DM-VTON
 * expects. Same algorithm as ml/vton/preprocess.py cloth_mask (not-white threshold, closing,
 * largest component, hole filling), so on-device and training-time masks agree.
 * Pure Java (unit-tested); pixels are ARGB, row-major.
 */
final class GarmentMask {

    /** Max-channel distance from pure white that counts as garment. */
    static final int WHITE_THRESHOLD = 30;

    private GarmentMask() {
    }

    /** Returns 1 for garment pixels, 0 for background. */
    static float[] compute(int[] argb, int w, int h) {
        boolean[] fg = new boolean[w * h];
        for (int i = 0; i < fg.length; i++) {
            int p = argb[i];
            int min = Math.min((p >> 16) & 0xFF, Math.min((p >> 8) & 0xFF, p & 0xFF));
            fg[i] = 255 - min > WHITE_THRESHOLD;
        }
        fg = erode(dilate(fg, w, h), w, h); // closing joins thin gaps
        fg = largestComponent(fg, w, h);
        fillHoles(fg, w, h);

        float[] out = new float[w * h];
        for (int i = 0; i < out.length; i++) {
            out[i] = fg[i] ? 1f : 0f;
        }
        return out;
    }

    /** 3x3 dilation; pixels outside the image don't contribute. */
    static boolean[] dilate(boolean[] m, int w, int h) {
        boolean[] out = new boolean[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean v = false;
                for (int dy = -1; dy <= 1 && !v; dy++) {
                    for (int dx = -1; dx <= 1 && !v; dx++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        v = yy >= 0 && yy < h && xx >= 0 && xx < w && m[yy * w + xx];
                    }
                }
                out[y * w + x] = v;
            }
        }
        return out;
    }

    /** 3x3 erosion; pixels outside the image don't remove anything. */
    static boolean[] erode(boolean[] m, int w, int h) {
        boolean[] out = new boolean[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean v = true;
                for (int dy = -1; dy <= 1 && v; dy++) {
                    for (int dx = -1; dx <= 1 && v; dx++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                            v = m[yy * w + xx];
                        }
                    }
                }
                out[y * w + x] = v;
            }
        }
        return out;
    }

    /** Keeps only the largest 8-connected foreground component. */
    static boolean[] largestComponent(boolean[] m, int w, int h) {
        int[] label = new int[w * h];
        int[] queue = new int[w * h];
        int best = 0;
        int bestSize = 0;
        int next = 0;
        for (int start = 0; start < m.length; start++) {
            if (!m[start] || label[start] != 0) {
                continue;
            }
            next++;
            int size = flood(m, true, label, next, start, queue, w, h);
            if (size > bestSize) {
                bestSize = size;
                best = next;
            }
        }
        boolean[] out = new boolean[w * h];
        for (int i = 0; i < out.length; i++) {
            out[i] = best != 0 && label[i] == best;
        }
        return out;
    }

    /** Sets background pixels not reachable from the image border (holes) to foreground. */
    static void fillHoles(boolean[] m, int w, int h) {
        int[] reached = new int[w * h];
        int[] queue = new int[w * h];
        for (int x = 0; x < w; x++) {
            seedBackground(m, reached, queue, x, 0, w, h);
            seedBackground(m, reached, queue, x, h - 1, w, h);
        }
        for (int y = 0; y < h; y++) {
            seedBackground(m, reached, queue, 0, y, w, h);
            seedBackground(m, reached, queue, w - 1, y, w, h);
        }
        for (int i = 0; i < m.length; i++) {
            if (!m[i] && reached[i] == 0) {
                m[i] = true;
            }
        }
    }

    private static void seedBackground(boolean[] m, int[] reached, int[] queue, int x, int y, int w, int h) {
        int i = y * w + x;
        if (!m[i] && reached[i] == 0) {
            flood(m, false, reached, 1, i, queue, w, h);
        }
    }

    /** 8-connected flood fill over pixels equal to {@code value}; returns the region size. */
    private static int flood(boolean[] m, boolean value, int[] label, int id, int start, int[] queue,
                             int w, int h) {
        int head = 0;
        int tail = 0;
        queue[tail++] = start;
        label[start] = id;
        while (head < tail) {
            int i = queue[head++];
            int x = i % w;
            int y = i / w;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int xx = x + dx;
                    int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        continue;
                    }
                    int j = yy * w + xx;
                    if (m[j] == value && label[j] == 0) {
                        label[j] = id;
                        queue[tail++] = j;
                    }
                }
            }
        }
        return tail;
    }
}
