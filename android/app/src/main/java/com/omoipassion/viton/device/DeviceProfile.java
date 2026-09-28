package com.omoipassion.viton.device;

public final class DeviceProfile {

    public final DeviceTier tier;
    /** Run models on the GPU: delegate works, output matches CPU, and it's faster. */
    public final boolean gpuUsable;
    public final long totalRamMb;
    public final boolean is64Bit;
    /** Reference-workload median latency; NaN if not measured (yet, or model missing). */
    public final float benchCpuMs;
    public final float benchGpuMs;

    public DeviceProfile(DeviceTier tier, boolean gpuUsable, long totalRamMb, boolean is64Bit,
                         float benchCpuMs, float benchGpuMs) {
        this.tier = tier;
        this.gpuUsable = gpuUsable;
        this.totalRamMb = totalRamMb;
        this.is64Bit = is64Bit;
        this.benchCpuMs = benchCpuMs;
        this.benchGpuMs = benchGpuMs;
    }

    public boolean benchmarked() {
        return !Float.isNaN(benchCpuMs);
    }

    /** Latency of the backend the pipeline will use; NaN if not benchmarked. */
    public float bestMs() {
        return gpuUsable && !Float.isNaN(benchGpuMs) ? benchGpuMs : benchCpuMs;
    }
}
