package com.omoipassion.viton.pipeline;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.Nullable;

import com.omoipassion.viton.device.DeviceProfile;
import com.omoipassion.viton.ml.LiteRtRunner;
import com.omoipassion.viton.ml.ModelSpec;
import com.omoipassion.viton.util.AppExecutors;
import com.omoipassion.viton.util.ImageOps;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * Person photo + garment photo → try-on image, fully on device.
 * <p>
 * Stages: pose check → crop to model aspect → model inference → composite back.
 * All model state lives on {@link AppExecutors#inference()}; public methods are safe to call
 * from the main thread and deliver callbacks there.
 */
public final class TryOnPipeline {

    public interface Callback {
        void onSuccess(TryOnResult result);

        void onError(TryOnException error);
    }

    private static final String TAG = "TryOnPipeline";

    private final Context appContext;
    private final DeviceProfile profile;

    // Owned by the inference thread.
    @Nullable
    private LiteRtRunner runner;
    @Nullable
    private ModelSpec spec;
    @Nullable
    private PoseChecker poseChecker;
    private boolean poseCheckerInitialized;
    private ByteBuffer personTensor;
    private ByteBuffer garmentTensor;
    private ByteBuffer outputTensor;

    public TryOnPipeline(Context context, DeviceProfile profile) {
        this.appContext = context.getApplicationContext();
        this.profile = profile;
    }

    public DeviceProfile profile() {
        return profile;
    }

    public void run(Bitmap person, Bitmap garment, Callback callback) {
        AppExecutors.inference().execute(() -> {
            try {
                TryOnResult result = runBlocking(person, garment);
                AppExecutors.runOnMain(() -> callback.onSuccess(result));
            } catch (TryOnException e) {
                AppExecutors.runOnMain(() -> callback.onError(e));
            } catch (OutOfMemoryError e) {
                // Realistic on 2-3 GB Android 8 devices; free the model and let the user retry.
                releaseNow();
                TryOnException err = new TryOnException(TryOnException.Reason.OUT_OF_MEMORY, e);
                AppExecutors.runOnMain(() -> callback.onError(err));
            } catch (IOException | RuntimeException | LinkageError e) {
                // Always call back, or the UI would wait forever.
                Log.e(TAG, "Try-on failed", e);
                TryOnException err = new TryOnException(TryOnException.Reason.INTERNAL, e);
                AppExecutors.runOnMain(() -> callback.onError(err));
            }
        });
    }

    /** Frees models and buffers; they are reloaded lazily on the next run. */
    public void release() {
        AppExecutors.inference().execute(this::releaseNow);
    }

    private TryOnResult runBlocking(Bitmap person, Bitmap garment)
            throws TryOnException, IOException {
        long start = SystemClock.elapsedRealtime();
        ensureModelLoaded();
        ModelSpec spec = this.spec;
        LiteRtRunner runner = this.runner;

        float centerX = 0.5f;
        PoseChecker pose = poseChecker();
        if (pose != null) {
            centerX = pose.check(person).centerX;
        }
        long poseMs = SystemClock.elapsedRealtime() - start;

        Rect crop = ImageOps.aspectCrop(person.getWidth(), person.getHeight(),
                centerX, spec.width, spec.height);
        // TODO(phase 2): feed a garment mask as a third input if the student model needs it.
        ImageOps.toTensor(ImageOps.cropAndScale(person, crop, spec.width, spec.height), personTensor);
        ImageOps.toTensor(ImageOps.fitCenter(garment, spec.width, spec.height), garmentTensor);

        long inferStart = SystemClock.elapsedRealtime();
        Map<Integer, Object> outputs = new HashMap<>();
        outputTensor.rewind();
        outputs.put(0, outputTensor);
        runner.run(new Object[]{personTensor, garmentTensor}, outputs);
        long inferenceMs = SystemClock.elapsedRealtime() - inferStart;

        Bitmap out = ImageOps.fromTensor(outputTensor, spec.width, spec.height);
        Bitmap composed = ImageOps.compose(person, crop, out);
        long totalMs = SystemClock.elapsedRealtime() - start;
        return new TryOnResult(composed, spec.name(), runner.isUsingGpu(),
                poseMs, inferenceMs, totalMs);
    }

    private void ensureModelLoaded() throws TryOnException, IOException {
        if (runner != null) {
            return;
        }
        ModelSpec resolved = ModelSpec.resolve(appContext, profile.tier);
        if (resolved == null) {
            throw new TryOnException(TryOnException.Reason.MODEL_MISSING);
        }
        LiteRtRunner r = LiteRtRunner.create(appContext, resolved.assetPath, profile.gpuUsable);
        try {
            int[] shape = {1, resolved.height, resolved.width, 3};
            r.checkInputShape(0, shape);
            r.checkInputShape(1, shape);
        } catch (RuntimeException e) {
            r.close();
            throw e;
        }
        Log.i(TAG, "Loaded " + resolved.name() + " on " + (r.isUsingGpu() ? "GPU" : "CPU")
                + " for tier " + profile.tier);
        spec = resolved;
        runner = r;
        personTensor = ImageOps.allocateRgbTensor(resolved.width, resolved.height);
        garmentTensor = ImageOps.allocateRgbTensor(resolved.width, resolved.height);
        outputTensor = ImageOps.allocateRgbTensor(resolved.width, resolved.height);
    }

    @Nullable
    private PoseChecker poseChecker() {
        if (!poseCheckerInitialized) {
            poseChecker = PoseChecker.createOrNull(appContext);
            poseCheckerInitialized = true;
        }
        return poseChecker;
    }

    private void releaseNow() {
        if (runner != null) {
            runner.close();
            runner = null;
        }
        if (poseChecker != null) {
            poseChecker.close();
            poseChecker = null;
        }
        poseCheckerInitialized = false;
        spec = null;
        personTensor = null;
        garmentTensor = null;
        outputTensor = null;
    }
}
