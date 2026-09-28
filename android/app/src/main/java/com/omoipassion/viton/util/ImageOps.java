package com.omoipassion.viton.util;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;

import java.nio.ByteBuffer;

/** Bitmap ⇄ tensor conversion and crop/composite helpers. Tensors are NHWC float32 in [-1, 1]. */
public final class ImageOps {

    private static final Paint FILTER = new Paint(Paint.FILTER_BITMAP_FLAG | Paint.ANTI_ALIAS_FLAG);

    private ImageOps() {
    }

    /**
     * Largest crop with the model's aspect ratio that fits the image, centered horizontally
     * on {@code centerX} (normalized 0..1, e.g. the torso center from pose detection).
     */
    public static Rect aspectCrop(int srcW, int srcH, float centerX, int dstW, int dstH) {
        float aspect = (float) dstW / dstH;
        int cropH = srcH;
        int cropW = Math.round(cropH * aspect);
        if (cropW > srcW) {
            cropW = srcW;
            cropH = Math.round(cropW / aspect);
        }
        int left = Math.round(centerX * srcW - cropW / 2f);
        left = Math.max(0, Math.min(left, srcW - cropW));
        int top = (srcH - cropH) / 2;
        return new Rect(left, top, left + cropW, top + cropH);
    }

    public static Bitmap cropAndScale(Bitmap src, Rect crop, int dstW, int dstH) {
        Bitmap out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888);
        new Canvas(out).drawBitmap(src, crop, new Rect(0, 0, dstW, dstH), FILTER);
        return out;
    }

    /** Letterboxes onto a white canvas, matching how garments appear in VITON-style datasets. */
    public static Bitmap fitCenter(Bitmap src, int dstW, int dstH) {
        Bitmap out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(out);
        c.drawColor(Color.WHITE);
        float s = Math.min((float) dstW / src.getWidth(), (float) dstH / src.getHeight());
        float w = src.getWidth() * s;
        float h = src.getHeight() * s;
        float l = (dstW - w) / 2f;
        float t = (dstH - h) / 2f;
        c.drawBitmap(src, null, new RectF(l, t, l + w, t + h), FILTER);
        return out;
    }

    public static ByteBuffer allocateRgbTensor(int w, int h) {
        return ByteBuffer.allocateDirect(4 * w * h * 3).order(java.nio.ByteOrder.nativeOrder());
    }

    public static void toTensor(Bitmap bmp, ByteBuffer out) {
        int w = bmp.getWidth();
        int h = bmp.getHeight();
        int[] px = new int[w * h];
        bmp.getPixels(px, 0, w, 0, 0, w, h);
        out.rewind();
        for (int p : px) {
            out.putFloat(((p >> 16) & 0xFF) / 127.5f - 1f);
            out.putFloat(((p >> 8) & 0xFF) / 127.5f - 1f);
            out.putFloat((p & 0xFF) / 127.5f - 1f);
        }
        out.rewind();
    }

    public static Bitmap fromTensor(ByteBuffer in, int w, int h) {
        in.rewind();
        int[] px = new int[w * h];
        for (int i = 0; i < px.length; i++) {
            int r = toByte(in.getFloat());
            int g = toByte(in.getFloat());
            int b = toByte(in.getFloat());
            px[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888);
    }

    /**
     * Pastes the model output back into the full-resolution original. Used when the
     * segmenter is unavailable; otherwise TryOnPipeline blends with a keep mask.
     */
    public static Bitmap compose(Bitmap original, Rect crop, Bitmap result) {
        Bitmap out = original.copy(Bitmap.Config.ARGB_8888, true);
        new Canvas(out).drawBitmap(result, null, crop, FILTER);
        return out;
    }

    private static int toByte(float v) {
        int x = Math.round((v + 1f) * 127.5f);
        return x < 0 ? 0 : Math.min(x, 255);
    }
}
