package com.omoipassion.viton.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Arrays;

public class GarmentMaskTest {

    private static final int WHITE = 0xFFFFFFFF;
    private static final int BLUE = 0xFF1E3CA0;
    private static final int BLACK = 0xFF000000;

    /** Same scene as ml/tests/test_preprocess.py, scaled down: garment with a white print. */
    @Test
    public void fillsPrintsAndDropsSpecks() {
        int w = 60;
        int h = 80;
        int[] img = new int[w * h];
        Arrays.fill(img, WHITE);
        fill(img, w, 16, 20, 44, 60, BLUE);   // garment: x 16..43, y 20..59
        fill(img, w, 26, 36, 34, 44, WHITE);  // white print inside it
        fill(img, w, 3, 3, 5, 5, BLACK);      // dust speck
        fill(img, w, 55, 74, 57, 76, BLACK);  // another speck

        float[] m = GarmentMask.compute(img, w, h);

        int wrong = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean expected = x >= 16 && x < 44 && y >= 20 && y < 60;
                if ((m[y * w + x] == 1f) != expected) {
                    wrong++;
                }
            }
        }
        assertEquals("pixels differing from the garment rectangle", 0, wrong);
        assertEquals(1f, m[40 * w + 30], 0f); // print is part of the garment
        assertEquals(0f, m[3 * w + 3], 0f);   // speck removed
    }

    @Test
    public void blankImageHasEmptyMask() {
        int[] img = new int[20 * 10];
        Arrays.fill(img, WHITE);
        for (float v : GarmentMask.compute(img, 10, 20)) {
            assertEquals(0f, v, 0f);
        }
    }

    @Test
    public void nearWhiteIsBackgroundButLightGreyIsGarment() {
        int w = 10;
        int h = 10;
        int[] img = new int[w * h];
        Arrays.fill(img, 0xFFF0F0F0);                 // 15 from white: background
        fill(img, w, 2, 2, 8, 8, 0xFFC8C8C8);         // 55 from white: garment
        float[] m = GarmentMask.compute(img, w, h);
        assertEquals(0f, m[0], 0f);
        assertEquals(1f, m[5 * w + 5], 0f);
    }

    @Test
    public void closingBridgesOnePixelGap() {
        int w = 12;
        int h = 6;
        boolean[] m = new boolean[w * h];
        for (int y = 1; y < 5; y++) {
            for (int x = 1; x < 11; x++) {
                m[y * w + x] = x != 6; // one-pixel vertical gap at x = 6
            }
        }
        boolean[] closed = GarmentMask.erode(GarmentMask.dilate(m, w, h), w, h);
        assertTrue(closed[2 * w + 6] && closed[3 * w + 6]);
    }

    private static void fill(int[] img, int w, int x0, int y0, int x1, int y1, int color) {
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                img[y * w + x] = color;
            }
        }
    }
}
