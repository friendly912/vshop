package com.omoipassion.viton.ui;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.omoipassion.viton.R;
import com.omoipassion.viton.databinding.ActivityCaptureBinding;

import java.io.File;

/** Takes the person photo with CameraX and returns its file path. */
public class CaptureActivity extends AppCompatActivity {

    public static final String EXTRA_PHOTO_PATH = "photo_path";

    private static final String TAG = "CaptureActivity";

    private ActivityCaptureBinding binding;
    private ProcessCameraProvider cameraProvider;
    private ImageCapture imageCapture;
    private CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;

    private final ActivityResultLauncher<String> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    startCamera();
                } else {
                    Toast.makeText(this, R.string.error_camera_permission, Toast.LENGTH_LONG).show();
                    finish();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCaptureBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.captureButton.setOnClickListener(v -> takePhoto());
        binding.switchButton.setOnClickListener(v -> switchCamera());

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindCamera();
            } catch (Exception e) {
                Log.e(TAG, "Camera init failed", e);
                Toast.makeText(this, R.string.error_camera, Toast.LENGTH_LONG).show();
                finish();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCamera() throws Exception {
        if (!cameraProvider.hasCamera(selector)) {
            selector = selector == CameraSelector.DEFAULT_BACK_CAMERA
                    ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;
        }
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());
        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build();
        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this, selector, preview, imageCapture);
    }

    private void switchCamera() {
        if (cameraProvider == null) {
            return;
        }
        selector = selector == CameraSelector.DEFAULT_BACK_CAMERA
                ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;
        try {
            bindCamera();
        } catch (Exception e) {
            Log.e(TAG, "Camera switch failed", e);
            Toast.makeText(this, R.string.error_camera, Toast.LENGTH_SHORT).show();
        }
    }

    private void takePhoto() {
        if (imageCapture == null) {
            return;
        }
        binding.captureButton.setEnabled(false);
        // The JPEG keeps its EXIF orientation; BitmapLoader applies it when decoding.
        File out = new File(getCacheDir(), "person_" + System.currentTimeMillis() + ".jpg");
        ImageCapture.OutputFileOptions options = new ImageCapture.OutputFileOptions.Builder(out).build();
        imageCapture.takePicture(options, ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults results) {
                        setResult(RESULT_OK,
                                new Intent().putExtra(EXTRA_PHOTO_PATH, out.getAbsolutePath()));
                        finish();
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException e) {
                        Log.e(TAG, "Capture failed", e);
                        Toast.makeText(CaptureActivity.this, R.string.error_capture,
                                Toast.LENGTH_SHORT).show();
                        binding.captureButton.setEnabled(true);
                    }
                });
    }
}
