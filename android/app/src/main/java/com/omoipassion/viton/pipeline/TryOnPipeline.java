package com.omoipassion.viton.pipeline;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.Nullable;

import com.omoipassion.viton.device.DeviceProfile;
import com.omoipassion.viton.device.DeviceTierClassifier;
import com.omoipassion.viton.ml.LiteRtRunner;
import com.omoipassion.viton.ml.ModelSpec;
import com.omoipassion.viton.util.AppExecutors;
import com.omoipassion.viton.util.ImageOps;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

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
    /** Replaced once by the first-run benchmark (on the inference thread). */
    private volatile DeviceProfile profile;

    // Owned by the inference thread.
    @Nullable
    private LiteRtRunner runner;
    @Nullable
    private ModelSpec spec;
    @Nullable
    private PoseChecker poseChecker;
    private boolean poseCheckerInitialized;
    @Nullable
    private PreserveSegmenter preserveSegmenter;
    private boolean preserveSegmenterInitialized;
    private ByteBuffer personTensor;
    private ByteBuffer garmentTensor;
    /** Only for models with a third (garment mask) input. */
    @Nullable
    private ByteBuffer garmentMaskTensor;
    private ByteBuffer outputTensor;

    public TryOnPipeline(Context context, DeviceProfile profile) {
        this.appContext = context.getApplicationContext();
        this.profile = profile;
        // First task on the single inference thread, so every try-on sees the measured tier.
        AppExecutors.inference().execute(() ->
                this.profile = DeviceTierClassifier.benchmarkIfNeeded(appContext, this.profile));
    }

    public DeviceProfile profile() {
        return profile;
    }

    /** Delivers the profile on the main thread once the first-run benchmark has finished. */
    public void whenProfileReady(Consumer<DeviceProfile> callback) {
        AppExecutors.inference().execute(() -> {
            DeviceProfile p = profile;
            AppExecutors.runOnMain(() -> callback.accept(p));
        });
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

        PoseChecker checker = poseChecker();
        PoseChecker.Pose pose = checker != null ? checker.check(person) : null;
        long poseMs = SystemClock.elapsedRealtime() - start;

        Rect crop = ImageOps.aspectCrop(person.getWidth(), person.getHeight(),
                pose != null ? pose.centerX : 0.5f, spec.width, spec.height);
        ImageOps.toTensor(ImageOps.cropAndScale(person, crop, spec.width, spec.height), personTensor);
        Bitmap garmentIn = ImageOps.fitCenter(garment, spec.width, spec.height);
        ImageOps.toTensor(garmentIn, garmentTensor);
        Object[] inputs;
        if (garmentMaskTensor != null) {
            int[] px = new int[spec.width * spec.height];
            garmentIn.getPixels(px, 0, spec.width, 0, 0, spec.width, spec.height);
            garmentMaskTensor.rewind();
            garmentMaskTensor.asFloatBuffer().put(GarmentMask.compute(px, spec.width, spec.height));
            inputs = new Object[]{personTensor, garmentTensor, garmentMaskTensor};
        } else {
            inputs = new Object[]{personTensor, garmentTensor};
        }

        long inferStart = SystemClock.elapsedRealtime();
        Map<Integer, Object> outputs = new HashMap<>();
        outputTensor.rewind();
        outputs.put(0, outputTensor);
        runner.run(inputs, outputs);
        long inferenceMs = SystemClock.elapsedRealtime() - inferStart;

        long keepStart = SystemClock.elapsedRealtime();
        PreserveSegmenter preserver = preserveSegmenter();
        PreserveSegmenter.KeepMask keep = preserver != null ? preserver.keepMask(person, pose) : null;
        long keepMs = SystemClock.elapsedRealtime() - keepStart;

        Bitmap out = ImageOps.fromTensor(outputTensor, spec.width, spec.height);
        Bitmap composed = compose(person, crop, out, keep);
        long totalMs = SystemClock.elapsedRealtime() - start;
        return new TryOnResult(composed, spec.name(), runner.isUsingGpu(),
                poseMs, inferenceMs, keepMs, totalMs);
    }

    /**
     * Pastes the model output into the crop of the full-resolution original, keeping the
     * original wherever {@code keep} says so (face, hair, hands, far background).
     */
    private static Bitmap compose(Bitmap original, Rect crop, Bitmap result,
                                  @Nullable PreserveSegmenter.KeepMask keep) {
        if (keep == null) {
            return ImageOps.compose(original, crop, result);
        }
        int cw = crop.width();
        int ch = crop.height();
        Bitmap scaled = Bitmap.createScaledBitmap(result, cw, ch, true);
        int[] tryOn = new int[cw * ch];
        int[] orig = new int[cw * ch];
        scaled.getPixels(tryOn, 0, cw, 0, 0, cw, ch);
        original.getPixels(orig, 0, cw, crop.left, crop.top, cw, ch);
        MaskOps.blendInto(tryOn, orig, cw, ch, crop.left, crop.top,
                original.getWidth(), original.getHeight(), keep.alpha, keep.width, keep.height);

        Bitmap out = original.copy(Bitmap.Config.ARGB_8888, true);
        out.setPixels(tryOn, 0, cw, crop.left, crop.top, cw, ch);
        return out;
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
            if (r.inputCount() == 3) { // e.g. DM-VTON: garment mask input
                r.checkInputShape(2, 1, resolved.height, resolved.width, 1);
            }
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
        garmentMaskTensor = r.inputCount() == 3
                ? ByteBuffer.allocateDirect(4 * resolved.width * resolved.height).order(ByteOrder.nativeOrder())
                : null;
    }

    @Nullable
    private PoseChecker poseChecker() {
        if (!poseCheckerInitialized) {
            poseChecker = PoseChecker.createOrNull(appContext);
            poseCheckerInitialized = true;
        }
        return poseChecker;
    }

    @Nullable
    private PreserveSegmenter preserveSegmenter() {
        if (!preserveSegmenterInitialized) {
            preserveSegmenter = PreserveSegmenter.createOrNull(appContext);
            preserveSegmenterInitialized = true;
        }
        return preserveSegmenter;
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
        if (preserveSegmenter != null) {
            preserveSegmenter.close();
            preserveSegmenter = null;
        }
        preserveSegmenterInitialized = false;
        spec = null;
        personTensor = null;
        garmentTensor = null;
        garmentMaskTensor = null;
        outputTensor = null;
    }
}
