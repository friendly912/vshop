package com.omoipassion.viton.ml;

import android.content.Context;

import androidx.annotation.Nullable;

import com.omoipassion.viton.device.DeviceTier;
import com.omoipassion.viton.util.Assets;

/** A bundled try-on model and its fixed input size. I/O contract: assets/models/README.md. */
public final class ModelSpec {

    public final String assetPath;
    public final int width;
    public final int height;

    public ModelSpec(String assetPath, int width, int height) {
        this.assetPath = assetPath;
        this.width = width;
        this.height = height;
    }

    public String name() {
        int slash = assetPath.lastIndexOf('/');
        return slash >= 0 ? assetPath.substring(slash + 1) : assetPath;
    }

    /**
     * The model for {@code tier}, or the next lighter tier's model if that one isn't
     * installed (for example, when only the Tier C model ships in the base APK).
     */
    @Nullable
    public static ModelSpec resolve(Context context, DeviceTier tier) {
        DeviceTier[] tiers = DeviceTier.values();
        for (int i = tier.ordinal(); i < tiers.length; i++) {
            if (Assets.exists(context, tiers[i].model.assetPath)) {
                return tiers[i].model;
            }
        }
        return null;
    }
}
