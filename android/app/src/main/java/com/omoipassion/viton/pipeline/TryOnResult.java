package com.omoipassion.viton.pipeline;

import android.graphics.Bitmap;

public final class TryOnResult {

    public final Bitmap image;
    public final String modelName;
    public final boolean usedGpu;
    public final long poseMs;
    public final long inferenceMs;
    /** Segmentation + keep-mask time (face/hair/hand preservation). */
    public final long keepMs;
    public final long totalMs;

    TryOnResult(Bitmap image, String modelName, boolean usedGpu,
                long poseMs, long inferenceMs, long keepMs, long totalMs) {
        this.image = image;
        this.modelName = modelName;
        this.usedGpu = usedGpu;
        this.poseMs = poseMs;
        this.inferenceMs = inferenceMs;
        this.keepMs = keepMs;
        this.totalMs = totalMs;
    }
}
