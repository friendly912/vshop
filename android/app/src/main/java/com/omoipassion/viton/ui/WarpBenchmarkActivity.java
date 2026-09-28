package com.omoipassion.viton.ui;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.omoipassion.viton.R;
import com.omoipassion.viton.VitonApp;
import com.omoipassion.viton.databinding.ActivityWarpBenchmarkBinding;
import com.omoipassion.viton.device.DeviceProfile;
import com.omoipassion.viton.ml.LiteRtRunner;
import com.omoipassion.viton.ml.WarpOps;
import com.omoipassion.viton.pipeline.TryOnPipeline;
import com.omoipassion.viton.util.AppExecutors;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Times the warp variants from ml/warp_spike on this device: each model on CPU and GPU,
 * GPU-vs-CPU output diff (catches driver bugs), and the plain Java warp (variant D).
 */
public class WarpBenchmarkActivity extends AppCompatActivity {

    /**
     * Starts the benchmark without a tap, for remote device farms. Launch via the exported
     * MainActivity: {@code adb shell am start -n com.omoipassion.viton/.ui.MainActivity
     * --ez warp_autorun true}, then read {@code adb logcat -s WarpBench} until {@link #DONE_MARKER}.
     */
    public static final String EXTRA_AUTORUN = "warp_autorun";
    public static final String DONE_MARKER = "WARP_BENCH_DONE";

    private static final String TAG = "WarpBench";
    private static final String DIR = "models/warp";
    private static final int WARMUP = 3;
    private static final int RUNS = 10;
    /** Kept below the smallest shift radius (2) so every variant is valid on the test grid. */
    private static final float MAX_SHIFT_PX = 1.5f;

    private ActivityWarpBenchmarkBinding binding;
    private String report = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityWarpBenchmarkBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        binding.runButton.setOnClickListener(v -> start());
        binding.copyButton.setOnClickListener(v -> copyReport());
        if (savedInstanceState == null && getIntent().getBooleanExtra(EXTRA_AUTORUN, false)) {
            start();
        }
    }

    private void start() {
        binding.runButton.setEnabled(false);
        binding.reportText.setText(R.string.bench_running);
        Context app = getApplicationContext();
        TryOnPipeline pipeline = ((VitonApp) getApplication()).pipeline();
        // Same thread as the try-on pipeline: the GPU delegate is thread-bound. Queued after
        // the first-run device benchmark, so the profile below is the measured one.
        AppExecutors.inference().execute(() -> {
            StringBuilder sb = new StringBuilder();
            line(sb, deviceHeader(pipeline.profile()));
            try {
                String[] files = app.getAssets().list(DIR);
                if (files == null || files.length == 0) {
                    line(sb, "No models in assets/" + DIR + ". Run ml/warp_spike/build_warp_models.py.");
                }
                Arrays.sort(files == null ? new String[0] : files);
                Set<String> javaTimed = new HashSet<>();
                for (String f : files == null ? new String[0] : files) {
                    if (f.endsWith(".tflite")) {
                        benchModel(app, DIR + "/" + f, f, sb, javaTimed);
                        publish(sb, false);
                    }
                }
            } catch (IOException | RuntimeException e) {
                Log.e(TAG, "Benchmark failed", e);
                line(sb, "FAILED: " + e);
            }
            line(sb, DONE_MARKER);
            publish(sb, true);
        });
    }

    private void benchModel(Context ctx, String path, String name, StringBuilder sb,
                            Set<String> javaTimed) throws IOException {
        line(sb, "\n" + name);
        LiteRtRunner cpu = LiteRtRunner.create(ctx, path, false);
        try {
            int imgIdx = cpu.inputShape(0)[3] == 3 ? 0 : 1;
            int gridIdx = 1 - imgIdx;
            int h = cpu.inputShape(imgIdx)[1];
            int w = cpu.inputShape(imgIdx)[2];

            float[] img = testImage(h, w);
            float[] grid = testGrid(h, w);
            Object[] inputs = new Object[2];
            inputs[imgIdx] = toBuffer(img);
            inputs[gridIdx] = toBuffer(grid);

            float[] ref = new float[h * w * 3];
            String size = h + "x" + w;
            if (javaTimed.add(size)) {
                for (int i = 0; i < WARMUP; i++) {
                    WarpOps.bilinear(img, grid, h, w, 3, ref);
                }
                long t0 = SystemClock.elapsedRealtimeNanos();
                for (int i = 0; i < RUNS; i++) {
                    WarpOps.bilinear(img, grid, h, w, 3, ref);
                }
                line(sb, String.format(Locale.US, "  java warp (D) %s: %.1f ms", size,
                        (SystemClock.elapsedRealtimeNanos() - t0) / 1e6 / RUNS));
            } else {
                WarpOps.bilinear(img, grid, h, w, 3, ref);
            }

            float[] cpuOut = new float[h * w * 3];
            double cpuMs = time(cpu, inputs, cpuOut);
            line(sb, String.format(Locale.US, "  CPU: %.1f ms, max err vs java %.1e",
                    cpuMs, maxDiff(cpuOut, ref)));

            LiteRtRunner gpu = LiteRtRunner.create(ctx, path, true);
            try {
                if (!gpu.isUsingGpu()) {
                    line(sb, "  GPU: REJECTED (delegate unavailable or init failed)");
                    return;
                }
                float[] gpuOut = new float[h * w * 3];
                double gpuMs = time(gpu, inputs, gpuOut);
                // fp16 on GPU: expect ~1e-3; much larger means a broken op/driver.
                line(sb, String.format(Locale.US, "  GPU: %.1f ms, max diff vs CPU %.1e",
                        gpuMs, maxDiff(gpuOut, cpuOut)));
            } finally {
                gpu.close();
            }
        } catch (RuntimeException e) {
            Log.e(TAG, name, e);
            line(sb, "  ERROR: " + e.getMessage());
        } finally {
            cpu.close();
        }
    }

    private static double time(LiteRtRunner runner, Object[] inputs, float[] result) {
        ByteBuffer out = ByteBuffer.allocateDirect(4 * result.length).order(ByteOrder.nativeOrder());
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, out);
        for (int i = 0; i < WARMUP; i++) {
            runOnce(runner, inputs, outputs, out);
        }
        long t0 = SystemClock.elapsedRealtimeNanos();
        for (int i = 0; i < RUNS; i++) {
            runOnce(runner, inputs, outputs, out);
        }
        double ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6 / RUNS;
        out.rewind();
        out.asFloatBuffer().get(result);
        return ms;
    }

    private static void runOnce(LiteRtRunner runner, Object[] inputs, Map<Integer, Object> outputs,
                                ByteBuffer out) {
        for (Object in : inputs) {
            ((ByteBuffer) in).rewind();
        }
        out.rewind();
        runner.run(inputs, outputs);
    }

    private static float[] testImage(int h, int w) {
        Random rnd = new Random(0);
        float[] img = new float[h * w * 3];
        for (int i = 0; i < img.length; i++) {
            img[i] = rnd.nextFloat() * 2f - 1f;
        }
        return img;
    }

    /** Identity grid plus a smooth displacement of at most MAX_SHIFT_PX pixels. */
    private static float[] testGrid(int h, int w) {
        float[] g = new float[h * w * 2];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float dx = MAX_SHIFT_PX * (float) Math.sin(4 * Math.PI * y / h);
                float dy = MAX_SHIFT_PX * (float) Math.cos(6 * Math.PI * x / w);
                int p = 2 * (y * w + x);
                g[p] = (2f * (x + dx) + 1f) / w - 1f;
                g[p + 1] = (2f * (y + dy) + 1f) / h - 1f;
            }
        }
        return g;
    }

    private static ByteBuffer toBuffer(float[] data) {
        ByteBuffer b = ByteBuffer.allocateDirect(4 * data.length).order(ByteOrder.nativeOrder());
        b.asFloatBuffer().put(data);
        return b;
    }

    private static float maxDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    private static String deviceHeader(DeviceProfile p) {
        String soc = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ? Build.SOC_MODEL : Build.HARDWARE;
        return String.format(Locale.US, "%s %s | Android %s (API %d) | SoC %s | %s | tier %s, RAM %d MB"
                        + "\nreference bench: CPU %.1f ms, GPU %.1f ms, use GPU %b",
                Build.MANUFACTURER, Build.MODEL, Build.VERSION.RELEASE, Build.VERSION.SDK_INT, soc,
                Build.SUPPORTED_ABIS[0], p.tier, p.totalRamMb, p.benchCpuMs, p.benchGpuMs, p.gpuUsable);
    }

    private static void line(StringBuilder sb, String s) {
        Log.i(TAG, s);
        sb.append(s).append('\n');
    }

    private void publish(StringBuilder sb, boolean finished) {
        String text = sb.toString();
        AppExecutors.runOnMain(() -> {
            if (isDestroyed()) {
                return;
            }
            report = text;
            binding.reportText.setText(text);
            if (finished) {
                binding.runButton.setEnabled(true);
            }
        });
    }

    private void copyReport() {
        ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("warp benchmark", report));
        Toast.makeText(this, R.string.bench_copied, Toast.LENGTH_SHORT).show();
    }
}
