package com.omoipassion.viton.device;

import android.app.ActivityManager;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import com.omoipassion.viton.ml.LiteRtRunner;
import com.omoipassion.viton.util.Assets;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Picks a {@link DeviceTier}. {@link #classify} is instant (cached profile, or a hardware
 * heuristic); {@link #benchmarkIfNeeded} then measures a reference workload once per OS/driver
 * build and replaces the guess with measured CPU/GPU speed.
 */
public final class DeviceTierClassifier {

    static final String BENCH_ASSET = "models/bench_ref.tflite";

    /**
     * Reference-workload latency limits per tier. Provisional: recalibrate from device reports
     * (logged as "DeviceTier") once the phase 2 student model exists.
     */
    static final float TIER_A_MAX_MS = 5f;
    static final float TIER_B_MAX_MS = 30f;
    /** Memory gates: tier A/B models need a 64-bit process and headroom. */
    static final long TIER_A_MIN_RAM_MB = 7000;
    static final long TIER_B_MIN_RAM_MB = 3500; // a "4 GB" phone reports about 3.6 GB
    /** GPU output may differ from CPU by fp16 rounding, not more. */
    static final float GPU_MAX_REL_ERROR = 0.05f;

    private static final String TAG = "DeviceTier";
    private static final String PREFS = "device_profile";
    /** Bump when tier logic changes so cached profiles are recomputed. */
    private static final int PROFILE_VERSION = 2;
    private static final int GLES_3_1 = 0x30001;
    private static final int WARMUP = 2;
    private static final int RUNS = 5;
    /** Stop timing early on very slow devices; one run is enough to know it's tier C. */
    private static final float SLOW_MS = 400f;

    private DeviceTierClassifier() {
    }

    /** Fast; safe on the main thread. */
    public static DeviceProfile classify(Context context) {
        SharedPreferences prefs = prefs(context);
        if (prefs.getInt("version", 0) == PROFILE_VERSION) {
            return new DeviceProfile(
                    DeviceTier.valueOf(prefs.getString("tier", DeviceTier.C.name())),
                    prefs.getBoolean("gpu", false),
                    prefs.getLong("ram_mb", 0),
                    prefs.getBoolean("is64", false),
                    prefs.getFloat("bench_cpu", Float.NaN),
                    prefs.getFloat("bench_gpu", Float.NaN));
        }
        Hardware hw = Hardware.read(context);
        // Unmeasured guess; the benchmark replaces it shortly.
        DeviceTier tier;
        if (!hw.is64 || hw.ramMb < TIER_B_MIN_RAM_MB || Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            tier = DeviceTier.C;
        } else if (hw.ramMb >= TIER_A_MIN_RAM_MB && hw.gles31
                && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            tier = DeviceTier.A;
        } else {
            tier = DeviceTier.B;
        }
        return new DeviceProfile(tier, hw.gles31, hw.ramMb, hw.is64, Float.NaN, Float.NaN);
    }

    /**
     * Blocking (about 1 s, longer on first GPU shader compile). Call on the inference thread.
     * Returns {@code current} unchanged if already measured on this OS build, or if the
     * benchmark model isn't installed.
     */
    public static DeviceProfile benchmarkIfNeeded(Context context, DeviceProfile current) {
        SharedPreferences prefs = prefs(context);
        if (current.benchmarked() && Build.FINGERPRINT.equals(prefs.getString("fingerprint", ""))) {
            return current;
        }
        if (!Assets.exists(context, BENCH_ASSET)) {
            Log.w(TAG, BENCH_ASSET + " not found; keeping heuristic tier " + current.tier);
            return current;
        }
        Hardware hw = Hardware.read(context);
        float cpuMs;
        float gpuMs = Float.NaN;
        boolean gpuOk = false;
        try (LiteRtRunner cpu = LiteRtRunner.create(context, BENCH_ASSET, false)) {
            ByteBuffer input = randomInput(cpu.inputShape(0));
            float[] cpuOut = new float[count(cpu.outputShape(0))];
            cpuMs = medianMs(cpu, input, cpuOut);

            if (hw.gles31) {
                try (LiteRtRunner gpu = LiteRtRunner.create(context, BENCH_ASSET, true)) {
                    if (gpu.isUsingGpu()) {
                        float[] gpuOut = new float[cpuOut.length];
                        gpuMs = medianMs(gpu, input, gpuOut);
                        gpuOk = gpuMatchesCpu(cpuOut, gpuOut);
                    }
                }
            }
        } catch (IOException | RuntimeException e) {
            Log.w(TAG, "Benchmark failed; keeping heuristic tier " + current.tier, e);
            return current;
        }

        boolean useGpu = gpuOk && gpuMs < cpuMs;
        float best = useGpu ? gpuMs : cpuMs;
        DeviceTier tier = decideTier(hw.ramMb, hw.is64, best);
        DeviceProfile p = new DeviceProfile(tier, useGpu, hw.ramMb, hw.is64, cpuMs, gpuMs);
        Log.i(TAG, String.format(java.util.Locale.US,
                "tier=%s cpu=%.1fms gpu=%.1fms gpuOk=%b useGpu=%b ram=%dMB is64=%b soc=%s model=%s",
                tier, cpuMs, gpuMs, gpuOk, useGpu, hw.ramMb, hw.is64, Build.HARDWARE, Build.MODEL));

        prefs.edit()
                .putInt("version", PROFILE_VERSION)
                .putString("fingerprint", Build.FINGERPRINT)
                .putString("tier", tier.name())
                .putBoolean("gpu", useGpu)
                .putLong("ram_mb", hw.ramMb)
                .putBoolean("is64", hw.is64)
                .putFloat("bench_cpu", cpuMs)
                .putFloat("bench_gpu", gpuMs)
                .apply();
        return p;
    }

    /** Memory gates first (they're hard limits), then measured speed. */
    static DeviceTier decideTier(long ramMb, boolean is64, float bestMs) {
        if (!is64 || ramMb < TIER_B_MIN_RAM_MB || Float.isNaN(bestMs)) {
            return DeviceTier.C;
        }
        if (bestMs <= TIER_A_MAX_MS && ramMb >= TIER_A_MIN_RAM_MB) {
            return DeviceTier.A;
        }
        return bestMs <= TIER_B_MAX_MS ? DeviceTier.B : DeviceTier.C;
    }

    /** False if the GPU result is wrong beyond fp16 rounding (driver bug) or not finite. */
    static boolean gpuMatchesCpu(float[] cpu, float[] gpu) {
        if (cpu.length != gpu.length) {
            return false;
        }
        float scale = 1f;
        float maxDiff = 0f;
        for (int i = 0; i < cpu.length; i++) {
            if (!Float.isFinite(gpu[i])) {
                return false;
            }
            scale = Math.max(scale, Math.abs(cpu[i]));
            maxDiff = Math.max(maxDiff, Math.abs(cpu[i] - gpu[i]));
        }
        return maxDiff <= GPU_MAX_REL_ERROR * scale;
    }

    private static float medianMs(LiteRtRunner runner, ByteBuffer input, float[] result) {
        ByteBuffer out = ByteBuffer.allocateDirect(4 * result.length).order(ByteOrder.nativeOrder());
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, out);
        Object[] inputs = {input};
        for (int i = 0; i < WARMUP; i++) {
            runOnce(runner, inputs, outputs, input, out);
        }
        float[] times = new float[RUNS];
        int n = 0;
        while (n < RUNS) {
            long t0 = SystemClock.elapsedRealtimeNanos();
            runOnce(runner, inputs, outputs, input, out);
            times[n++] = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f;
            if (times[n - 1] > SLOW_MS) {
                break;
            }
        }
        out.rewind();
        out.asFloatBuffer().get(result);
        float[] measured = Arrays.copyOf(times, n);
        Arrays.sort(measured);
        return measured[n / 2];
    }

    private static void runOnce(LiteRtRunner runner, Object[] inputs, Map<Integer, Object> outputs,
                                ByteBuffer input, ByteBuffer out) {
        input.rewind();
        out.rewind();
        runner.run(inputs, outputs);
    }

    private static ByteBuffer randomInput(int[] shape) {
        int n = count(shape);
        ByteBuffer b = ByteBuffer.allocateDirect(4 * n).order(ByteOrder.nativeOrder());
        Random rnd = new Random(0);
        for (int i = 0; i < n; i++) {
            b.putFloat(rnd.nextFloat() * 2f - 1f);
        }
        b.rewind();
        return b;
    }

    private static int count(int[] shape) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    private static SharedPreferences prefs(Context context) {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    private static final class Hardware {
        final long ramMb;
        final boolean is64;
        final boolean gles31;

        private Hardware(long ramMb, boolean is64, boolean gles31) {
            this.ramMb = ramMb;
            this.is64 = is64;
            this.gles31 = gles31;
        }

        static Hardware read(Context context) {
            ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo mem = new ActivityManager.MemoryInfo();
            am.getMemoryInfo(mem);
            return new Hardware(
                    mem.totalMem / (1024 * 1024),
                    Build.SUPPORTED_64_BIT_ABIS.length > 0,
                    // The GPU delegate needs OpenGL ES 3.1 compute; many Android 8 devices only have 3.0.
                    am.getDeviceConfigurationInfo().reqGlEsVersion >= GLES_3_1);
        }
    }
}
