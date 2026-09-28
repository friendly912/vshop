package com.omoipassion.viton.device;

public final class DeviceProfile {

    public final DeviceTier tier;
    public final boolean gpuUsable;
    public final long totalRamMb;
    public final boolean is64Bit;

    public DeviceProfile(DeviceTier tier, boolean gpuUsable, long totalRamMb, boolean is64Bit) {
        this.tier = tier;
        this.gpuUsable = gpuUsable;
        this.totalRamMb = totalRamMb;
        this.is64Bit = is64Bit;
    }
}
