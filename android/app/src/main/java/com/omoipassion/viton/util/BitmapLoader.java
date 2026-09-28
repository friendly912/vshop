package com.omoipassion.viton.util;

import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;

import androidx.exifinterface.media.ExifInterface;

import java.io.IOException;
import java.io.InputStream;

/** Decodes images down-sampled to a max dimension, with EXIF rotation applied. */
public final class BitmapLoader {

    private BitmapLoader() {
    }

    public static Bitmap decode(Context context, Uri uri, int maxDim) throws IOException {
        ContentResolver resolver = context.getContentResolver();

        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = open(resolver, uri)) {
            BitmapFactory.decodeStream(in, null, bounds);
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IOException("Not an image: " + uri);
        }

        // Power-of-two sampling keeps peak memory low on 2-3 GB devices.
        int sample = 1;
        while (Math.max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxDim) {
            sample *= 2;
        }
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = sample;
        opts.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Bitmap bmp;
        try (InputStream in = open(resolver, uri)) {
            bmp = BitmapFactory.decodeStream(in, null, opts);
        }
        if (bmp == null) {
            throw new IOException("Decode failed: " + uri);
        }

        int rotation;
        try (InputStream in = open(resolver, uri)) {
            rotation = new ExifInterface(in).getRotationDegrees();
        }
        Bitmap rotated = rotation == 0 ? bmp : rotate(bmp, rotation);
        return scaleDown(rotated, maxDim);
    }

    private static InputStream open(ContentResolver resolver, Uri uri) throws IOException {
        InputStream in = resolver.openInputStream(uri);
        if (in == null) {
            throw new IOException("Cannot open " + uri);
        }
        return in;
    }

    private static Bitmap rotate(Bitmap src, int degrees) {
        Matrix m = new Matrix();
        m.postRotate(degrees);
        Bitmap out = Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
        if (out != src) {
            src.recycle();
        }
        return out;
    }

    private static Bitmap scaleDown(Bitmap src, int maxDim) {
        int longest = Math.max(src.getWidth(), src.getHeight());
        if (longest <= maxDim) {
            return src;
        }
        float s = (float) maxDim / longest;
        Bitmap out = Bitmap.createScaledBitmap(src,
                Math.round(src.getWidth() * s), Math.round(src.getHeight() * s), true);
        if (out != src) {
            src.recycle();
        }
        return out;
    }
}
