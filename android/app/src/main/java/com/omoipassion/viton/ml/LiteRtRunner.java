package com.omoipassion.viton.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import androidx.annotation.Nullable;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Map;

/**
 * LiteRT interpreter with GPU delegate and automatic CPU (XNNPACK) fallback.
 * Not thread-safe: create, run and close on the same thread (see AppExecutors.inference()).
 */
public final class LiteRtRunner implements Closeable {

    private static final String TAG = "LiteRtRunner";

    private final Interpreter interpreter;
    @Nullable
    private final GpuDelegate gpuDelegate;

    private LiteRtRunner(Interpreter interpreter, @Nullable GpuDelegate gpuDelegate) {
        this.interpreter = interpreter;
        this.gpuDelegate = gpuDelegate;
    }

    public static LiteRtRunner create(Context context, String assetPath, boolean preferGpu)
            throws IOException {
        MappedByteBuffer model = mapAsset(context, assetPath);

        if (preferGpu) {
            CompatibilityList compat = new CompatibilityList();
            try {
                if (compat.isDelegateSupportedOnThisDevice()) {
                    GpuDelegate delegate = null;
                    try {
                        GpuDelegateFactory.Options gpuOpts = compat.getBestOptionsForThisDevice();
                        gpuOpts.setPrecisionLossAllowed(true); // fp16
                        delegate = new GpuDelegate(gpuOpts);
                        Interpreter.Options opts = new Interpreter.Options().addDelegate(delegate);
                        return new LiteRtRunner(new Interpreter(model, opts), delegate);
                    } catch (RuntimeException e) {
                        // Old drivers (Adreno 5xx, Mali-G71) often reject some ops at init.
                        Log.w(TAG, "GPU delegate failed, falling back to CPU", e);
                        if (delegate != null) {
                            delegate.close();
                        }
                    }
                }
            } finally {
                compat.close();
            }
        }

        Interpreter.Options opts = new Interpreter.Options()
                .setNumThreads(cpuThreads())
                .setUseXNNPACK(true);
        return new LiteRtRunner(new Interpreter(model, opts), null);
    }

    public boolean isUsingGpu() {
        return gpuDelegate != null;
    }

    /** Fails fast with a readable message if the model doesn't match the expected NHWC shape. */
    public void checkInputShape(int index, int... expected) {
        Tensor t = interpreter.getInputTensor(index);
        if (!Arrays.equals(t.shape(), expected)) {
            throw new IllegalStateException("Input " + index + " shape " + Arrays.toString(t.shape())
                    + " != expected " + Arrays.toString(expected));
        }
    }

    public int inputCount() {
        return interpreter.getInputTensorCount();
    }

    public int[] inputShape(int index) {
        return interpreter.getInputTensor(index).shape();
    }

    public int[] outputShape(int index) {
        return interpreter.getOutputTensor(index).shape();
    }

    public void run(Object[] inputs, Map<Integer, Object> outputs) {
        interpreter.runForMultipleInputsOutputs(inputs, outputs);
    }

    @Override
    public void close() {
        interpreter.close();
        if (gpuDelegate != null) {
            gpuDelegate.close();
        }
    }

    private static int cpuThreads() {
        // Roughly the big-core count on typical big.LITTLE SoCs.
        return Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors() / 2));
    }

    private static MappedByteBuffer mapAsset(Context context, String path) throws IOException {
        // openFd requires the asset to be stored uncompressed (see noCompress in app/build.gradle).
        try (AssetFileDescriptor fd = context.getAssets().openFd(path);
             FileInputStream in = new FileInputStream(fd.getFileDescriptor())) {
            return in.getChannel().map(
                    FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
        }
    }
}
