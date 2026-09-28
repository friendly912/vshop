package com.omoipassion.viton.device;

import android.app.ActivityManager;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;

/**
 * Picks a {@link DeviceTier} from static hardware signals and caches it.
 * TODO(phase 4): refine with a ~1 s on-device micro-benchmark on first launch.
 */
public final class DeviceTierClassifier {

    private static final String PREFS = "device_profile";
    /** Bump when the heuristic changes so cached profiles are recomputed. */
    private static final int PROFILE_VERSION = 1;
    private static final int GLES_3_1 = 0x30001;

    private DeviceTierClassifier() {
    }

    public static DeviceProfile classify(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        if (prefs.getInt("version", 0) == PROFILE_VERSION) {
            return new DeviceProfile(
                    DeviceTier.valueOf(prefs.getString("tier", DeviceTier.C.name())),
                    prefs.getBoolean("gpu", false),
                    prefs.getLong("ram_mb", 0),
                    prefs.getBoolean("is64", false));
        }

        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo mem = new ActivityManager.MemoryInfo();
        am.getMemoryInfo(mem);
        long ramMb = mem.totalMem / (1024 * 1024);
        boolean is64 = Build.SUPPORTED_64_BIT_ABIS.length > 0;
        // The GPU delegate needs OpenGL ES 3.1 compute; many Android 8 devices only have 3.0.
        boolean gles31 = am.getDeviceConfigurationInfo().reqGlEsVersion >= GLES_3_1;

        DeviceTier tier;
        // totalMem under-reports physical RAM: a "4 GB" phone reports roughly 3.6 GB.
        if (!is64 || ramMb < 3500 || Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            tier = DeviceTier.C;
        } else if (ramMb >= 7000 && gles31 && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            tier = DeviceTier.A;
        } else {
            tier = DeviceTier.B;
        }

        prefs.edit()
                .putInt("version", PROFILE_VERSION)
                .putString("tier", tier.name())
                .putBoolean("gpu", gles31)
                .putLong("ram_mb", ramMb)
                .putBoolean("is64", is64)
                .apply();
        return new DeviceProfile(tier, gles31, ramMb, is64);
    }
}
