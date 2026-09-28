package com.omoipassion.viton.pipeline;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.Nullable;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult;
import com.omoipassion.viton.util.Assets;

import java.io.Closeable;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Decides which pixels of the original photo survive the try-on (face, hair, accessories,
 * hands, far background) using MediaPipe's multiclass selfie segmenter.
 */
final class PreserveSegmenter implements Closeable {

    static final String MODEL_ASSET = "models/selfie_multiclass_256x256.tflite";

    /** Keep mask covering the whole person image, at mask resolution. */
    static final class KeepMask {
        final float[] alpha;
        final int width;
        final int height;

        KeepMask(float[] alpha, int width, int height) {
            this.alpha = alpha;
            this.width = width;
            this.height = height;
        }
    }

    private static final String TAG = "PreserveSegmenter";
    // Category order from the model's embedded labels.txt.
    private static final int BACKGROUND = 0;
    private static final int HAIR = 1;
    private static final int BODY_SKIN = 2;
    private static final int FACE_SKIN = 3;
    private static final int OTHERS = 5;
    /** Segmenter input size; masks come back at this size, keeping memory small. */
    private static final int MAX_DIM = 512;
    /** Background this far from the person (fraction of mask width) keeps the original. */
    private static final float BG_MARGIN = 0.06f;
    private static final int FEATHER_PX = 2;

    private final ImageSegmenter segmenter;

    private PreserveSegmenter(ImageSegmenter segmenter) {
        this.segmenter = segmenter;
    }

    /** Returns null (no preservation) if the model is missing or fails to load. */
    @Nullable
    static PreserveSegmenter createOrNull(Context context) {
        if (!Assets.exists(context, MODEL_ASSET)) {
            Log.w(TAG, MODEL_ASSET + " not found; face/hand preservation disabled");
            return null;
        }
        try {
            ImageSegmenter.ImageSegmenterOptions options =
                    ImageSegmenter.ImageSegmenterOptions.builder()
                            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_ASSET).build())
                            .setRunningMode(RunningMode.IMAGE)
                            .setOutputConfidenceMasks(true)
                            .setOutputCategoryMask(false)
                            .build();
            return new PreserveSegmenter(ImageSegmenter.createFromOptions(context, options));
        } catch (RuntimeException | LinkageError e) {
            Log.w(TAG, "Segmenter init failed; face/hand preservation disabled", e);
            return null;
        }
    }

    KeepMask keepMask(Bitmap person, @Nullable PoseChecker.Pose pose) {
        float s = Math.min(1f, (float) MAX_DIM / Math.max(person.getWidth(), person.getHeight()));
        Bitmap small = s < 1f
                ? Bitmap.createScaledBitmap(person, Math.round(person.getWidth() * s),
                Math.round(person.getHeight() * s), true)
                : person;

        ImageSegmenterResult result = segmenter.segment(new BitmapImageBuilder(small).build());
        List<MPImage> masks = result.confidenceMasks().orElseThrow(
                () -> new IllegalStateException("No confidence masks"));
        int w = masks.get(0).getWidth();
        int h = masks.get(0).getHeight();
        float[][] conf = new float[masks.size()][];
        for (int c = 0; c < masks.size(); c++) {
            MPImage m = masks.get(c);
            conf[c] = new float[w * h];
            ByteBufferExtractor.extract(m).order(ByteOrder.nativeOrder()).asFloatBuffer().get(conf[c]);
            m.close();
        }

        List<MaskOps.Circle> hands = new ArrayList<>(2);
        if (pose != null) {
            for (float[] hand : pose.hands) {
                hands.add(new MaskOps.Circle(hand[0] * w, hand[1] * h, hand[2] * w));
            }
        }
        float[] keep = MaskOps.keepMask(conf[BACKGROUND], conf[HAIR], conf[BODY_SKIN],
                conf[FACE_SKIN], conf[OTHERS], w, h, hands, Math.round(BG_MARGIN * w));
        return new KeepMask(MaskOps.boxBlur(keep, w, h, FEATHER_PX), w, h);
    }

    @Override
    public void close() {
        segmenter.close();
    }
}
