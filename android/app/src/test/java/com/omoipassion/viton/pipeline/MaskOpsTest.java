package com.omoipassion.viton.pipeline;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MaskOpsTest {

    private static final float EPS = 1e-5f;

    private static float[] filled(int n, float v) {
        float[] a = new float[n];
        Arrays.fill(a, v);
        return a;
    }

    @Test
    public void dilateGrowsSinglePixelToSquare() {
        int w = 9;
        int h = 9;
        float[] m = new float[w * h];
        m[4 * w + 4] = 1f;
        float[] d = MaskOps.dilate(m, w, h, 2);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean inside = Math.abs(x - 4) <= 2 && Math.abs(y - 4) <= 2;
                assertEquals("(" + x + "," + y + ")", inside ? 1f : 0f, d[y * w + x], EPS);
            }
        }
    }

    @Test
    public void boxBlurKeepsConstantAndSpreadsImpulse() {
        int w = 7;
        int h = 7;
        float[] c = MaskOps.boxBlur(filled(w * h, 0.3f), w, h, 2);
        for (float v : c) {
            assertEquals(0.3f, v, EPS);
        }
        float[] m = new float[w * h];
        m[3 * w + 3] = 1f;
        float[] b = MaskOps.boxBlur(m, w, h, 1);
        assertEquals(1f / 9f, b[3 * w + 3], EPS);
        assertEquals(1f / 9f, b[2 * w + 2], EPS);
        assertEquals(0f, b[0], EPS);
        float sum = 0f;
        for (float v : b) {
            sum += v;
        }
        assertEquals(1f, sum, EPS); // impulse far from edges: mass preserved
    }

    /**
     * 20x10 scene: person occupies columns 5..14; within it, rows 0..2 face, rows 3..9 clothes
     * except body skin at (6,8) and (13,8). Background elsewhere.
     */
    @Test
    public void keepMaskKeepsFaceHandsFarBackgroundButNotClothes() {
        int w = 20;
        int h = 10;
        int n = w * h;
        float[] bg = new float[n];
        float[] hair = new float[n];
        float[] body = new float[n];
        float[] face = new float[n];
        float[] others = new float[n];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                if (x < 5 || x > 14) {
                    bg[i] = 1f;
                } else if (y <= 2) {
                    face[i] = 1f;
                }
            }
        }
        body[8 * w + 6] = 1f;
        body[8 * w + 13] = 1f;
        // Only the left hand has a circle; the right body-skin pixel is e.g. a bare forearm.
        List<MaskOps.Circle> hands = Collections.singletonList(new MaskOps.Circle(6.5f, 8.5f, 1.5f));

        float[] k = MaskOps.keepMask(bg, hair, body, face, others, w, h, hands, 2);

        assertEquals("face", 1f, k[1 * w + 8], EPS);
        assertEquals("clothes", 0f, k[6 * w + 9], EPS);
        assertEquals("hand in circle", 1f, k[8 * w + 6], EPS);
        assertEquals("body skin outside circle", 0f, k[8 * w + 13], EPS);
        assertEquals("far background", 1f, k[5 * w + 0], EPS);
        assertEquals("background within margin", 0f, k[5 * w + 3], EPS);
        assertEquals("background within margin (right)", 0f, k[5 * w + 16], EPS);
        assertEquals("far background (right)", 1f, k[5 * w + 19], EPS);
    }

    @Test
    public void blendIntoRespectsAlphaAndCropOffset() {
        // Full image 4x2, mask 4x2: left half keep (1), right half replace (0).
        float[] mask = {1f, 1f, 0f, 0f, 1f, 1f, 0f, 0f};
        int red = 0xFFFF0000;
        int blue = 0xFF0000FF;

        // Crop = columns 1..2 of row 0..1 (2x2); original red, try-on blue.
        int[] tryOn = {blue, blue, blue, blue};
        int[] orig = {red, red, red, red};
        MaskOps.blendInto(tryOn, orig, 2, 2, 1, 0, 4, 2, mask, 4, 2);
        assertEquals(red, tryOn[0]);   // column 1 -> keep original
        assertEquals(blue, tryOn[1]);  // column 2 -> try-on
        assertEquals(red, tryOn[2]);
        assertEquals(blue, tryOn[3]);
    }

    @Test
    public void blendIntoMixesHalfAlpha() {
        float[] mask = filled(4, 0.5f);
        int[] tryOn = {0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000};
        int[] orig = {0xFFC86432, 0xFFC86432, 0xFFC86432, 0xFFC86432}; // (200,100,50)
        MaskOps.blendInto(tryOn, orig, 2, 2, 0, 0, 2, 2, mask, 2, 2);
        assertEquals(0xFF643219, tryOn[0]); // (100,50,25)
    }

    @Test
    public void blendIntoSamplesSmallMaskAtFullResolution() {
        // Mask 2x1 (keep, replace) stretched over a 4x1 image: bilinear ramp in the middle.
        float[] mask = {1f, 0f};
        int[] tryOn = new int[4];
        int[] orig = new int[4];
        Arrays.fill(tryOn, 0xFF000000);
        Arrays.fill(orig, 0xFFFFFFFF);
        MaskOps.blendInto(tryOn, orig, 4, 1, 0, 0, 4, 1, mask, 2, 1);
        assertEquals(0xFFFFFFFF, tryOn[0]);           // clamped to mask[0] = 1
        assertEquals(0xFF000000, tryOn[3]);           // clamped to mask[1] = 0
        int g1 = (tryOn[1] >> 8) & 0xFF;
        int g2 = (tryOn[2] >> 8) & 0xFF;
        assertEquals(191, g1);                        // u = 0.25 -> a = 0.75
        assertEquals(64, g2);                         // u = 0.75 -> a = 0.25
    }
}
