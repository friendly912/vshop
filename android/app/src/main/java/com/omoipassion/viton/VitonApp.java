package com.omoipassion.viton;

import android.app.Application;

import com.omoipassion.viton.device.DeviceTierClassifier;
import com.omoipassion.viton.pipeline.TryOnPipeline;

public class VitonApp extends Application {

    private TryOnPipeline pipeline;

    /** The app-wide pipeline; models load lazily on the first try-on. */
    public synchronized TryOnPipeline pipeline() {
        if (pipeline == null) {
            pipeline = new TryOnPipeline(this, DeviceTierClassifier.classify(this));
        }
        return pipeline;
    }

    @SuppressWarnings("deprecation") // TRIM_MEMORY_BACKGROUND is still delivered on API 26-33.
    @Override
    public void onTrimMemory(int level) {
        super.onTrimMemory(level);
        if (level >= TRIM_MEMORY_BACKGROUND) {
            releasePipeline();
        }
    }

    @Override
    public void onLowMemory() {
        super.onLowMemory();
        releasePipeline();
    }

    private synchronized void releasePipeline() {
        if (pipeline != null) {
            pipeline.release();
        }
    }
}
