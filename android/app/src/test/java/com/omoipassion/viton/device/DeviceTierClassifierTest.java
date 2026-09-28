package com.omoipassion.viton.device;

import static com.omoipassion.viton.device.DeviceTierClassifier.TIER_A_MAX_MS;
import static com.omoipassion.viton.device.DeviceTierClassifier.TIER_B_MAX_MS;
import static com.omoipassion.viton.device.DeviceTierClassifier.decideTier;
import static com.omoipassion.viton.device.DeviceTierClassifier.gpuMatchesCpu;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class DeviceTierClassifierTest {

    @Test
    public void fastDeviceWithEnoughRamIsTierA() {
        assertEquals(DeviceTier.A, decideTier(8000, true, TIER_A_MAX_MS));
    }

    @Test
    public void fastDeviceWithoutRamForTierAIsTierB() {
        assertEquals(DeviceTier.B, decideTier(5500, true, 1f));
    }

    @Test
    public void midSpeedIsTierBRegardlessOfAndroidVersion() {
        assertEquals(DeviceTier.B, decideTier(8000, true, TIER_A_MAX_MS + 0.1f));
        assertEquals(DeviceTier.B, decideTier(4000, true, TIER_B_MAX_MS));
    }

    @Test
    public void slowDeviceIsTierC() {
        assertEquals(DeviceTier.C, decideTier(12000, true, TIER_B_MAX_MS + 0.1f));
    }

    @Test
    public void memoryGatesOverrideSpeed() {
        assertEquals("32-bit", DeviceTier.C, decideTier(8000, false, 1f));
        assertEquals("3 GB phone", DeviceTier.C, decideTier(2800, true, 1f));
    }

    @Test
    public void unmeasuredIsTierC() {
        assertEquals(DeviceTier.C, decideTier(8000, true, Float.NaN));
    }

    @Test
    public void gpuWithinFp16RoundingMatches() {
        float[] cpu = {0f, 1f, 6f, 3.25f};
        float[] gpu = {0.001f, 0.999f, 6.003f, 3.25f};
        assertTrue(gpuMatchesCpu(cpu, gpu));
    }

    @Test
    public void wrongGpuOutputIsRejected() {
        float[] cpu = {0f, 1f, 6f, 3.25f};
        assertFalse("garbage", gpuMatchesCpu(cpu, new float[]{0f, 1f, 0f, 3.25f}));
        assertFalse("NaN", gpuMatchesCpu(cpu, new float[]{0f, Float.NaN, 6f, 3.25f}));
        assertFalse("infinite", gpuMatchesCpu(cpu, new float[]{0f, 1f, Float.POSITIVE_INFINITY, 3.25f}));
        assertFalse("length", gpuMatchesCpu(cpu, new float[]{0f, 1f}));
    }

    @Test
    public void allZeroOutputUsesAbsoluteTolerance() {
        float[] cpu = new float[8];
        float[] gpu = new float[8];
        gpu[3] = 0.04f;
        assertTrue(gpuMatchesCpu(cpu, gpu));
        gpu[3] = 0.2f;
        assertFalse(gpuMatchesCpu(cpu, gpu));
    }
}
