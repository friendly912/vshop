package com.omoipassion.viton.pipeline;

import android.graphics.Bitmap;

public final class TryOnResult {

    public final Bitmap image;
    public final String modelName;
    public final boolean usedGpu;
    public final long poseMs;
    public final long inferenceMs;
    public final long totalMs;

    TryOnResult(Bitmap image, String modelName, boolean usedGpu,
                long poseMs, long inferenceMs, long totalMs) {
        this.image = image;
        this.modelName = modelName;
        this.usedGpu = usedGpu;
        this.poseMs = poseMs;
        this.inferenceMs = inferenceMs;
        this.totalMs = totalMs;
    }
}
