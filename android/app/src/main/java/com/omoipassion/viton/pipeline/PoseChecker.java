package com.omoipassion.viton.pipeline;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.Nullable;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import com.omoipassion.viton.util.Assets;

import java.io.Closeable;
import java.util.List;

/** Validates the person photo and finds the torso center, using MediaPipe Pose Landmarker. */
final class PoseChecker implements Closeable {

    static final String MODEL_ASSET = "models/pose_landmarker_lite.task";

    private static final String TAG = "PoseChecker";
    private static final int L_SHOULDER = 11;
    private static final int R_SHOULDER = 12;
    private static final int L_HIP = 23;
    private static final int R_HIP = 24;
    private static final int[] TORSO = {L_SHOULDER, R_SHOULDER, L_HIP, R_HIP};
    private static final float MIN_VISIBILITY = 0.5f;

    /** Torso center, normalized 0..1. */
    static final class Pose {
        final float centerX;

        Pose(float centerX) {
            this.centerX = centerX;
        }
    }

    private final PoseLandmarker landmarker;

    private PoseChecker(PoseLandmarker landmarker) {
        this.landmarker = landmarker;
    }

    /** Returns null (pose check skipped) if the model asset is missing or fails to load. */
    @Nullable
    static PoseChecker createOrNull(Context context) {
        if (!Assets.exists(context, MODEL_ASSET)) {
            Log.w(TAG, MODEL_ASSET + " not found; pose check disabled");
            return null;
        }
        try {
            PoseLandmarker.PoseLandmarkerOptions options =
                    PoseLandmarker.PoseLandmarkerOptions.builder()
                            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_ASSET).build())
                            .setRunningMode(RunningMode.IMAGE)
                            .setNumPoses(2) // detect 2 so multi-person photos can be rejected
                            .build();
            return new PoseChecker(PoseLandmarker.createFromOptions(context, options));
        } catch (RuntimeException | LinkageError e) {
            // LinkageError: MediaPipe ships no x86_64 native lib, so emulators land here.
            Log.w(TAG, "Pose landmarker init failed; pose check disabled", e);
            return null;
        }
    }

    Pose check(Bitmap person) throws TryOnException {
        PoseLandmarkerResult result = landmarker.detect(new BitmapImageBuilder(person).build());
        List<List<NormalizedLandmark>> poses = result.landmarks();
        if (poses.size() != 1) {
            throw new TryOnException(TryOnException.Reason.NO_PERSON);
        }
        List<NormalizedLandmark> lm = poses.get(0);
        float sumX = 0f;
        for (int i : TORSO) {
            NormalizedLandmark p = lm.get(i);
            if (p.visibility().orElse(0f) < MIN_VISIBILITY) {
                throw new TryOnException(TryOnException.Reason.TORSO_NOT_VISIBLE);
            }
            sumX += p.x();
        }
        return new Pose(sumX / TORSO.length);
    }

    @Override
    public void close() {
        landmarker.close();
    }
}
