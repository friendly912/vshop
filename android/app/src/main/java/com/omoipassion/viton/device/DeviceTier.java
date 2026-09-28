package com.omoipassion.viton.device;

import com.omoipassion.viton.ml.ModelSpec;

/** Capability tiers, best first. Each tier maps to one model build (see docs/ROADMAP.md). */
public enum DeviceTier {
    /** Flagships (8+ GB RAM, Android 12+): 512x384 model, GPU/NPU. */
    A(new ModelSpec("models/viton_a.tflite", 384, 512)),
    /** Mid-range 64-bit devices: 512x384 model, GPU. */
    B(new ModelSpec("models/viton_b.tflite", 384, 512)),
    /** Android 8-9 era, 32-bit or under 4 GB RAM: 256x192 micro model. */
    C(new ModelSpec("models/viton_c.tflite", 192, 256));

    public final ModelSpec model;

    DeviceTier(ModelSpec model) {
        this.model = model;
    }
}
