package com.omoipassion.viton.ui;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.os.BundleCompat;

import com.omoipassion.viton.R;
import com.omoipassion.viton.VitonApp;
import com.omoipassion.viton.databinding.ActivityMainBinding;
import com.omoipassion.viton.device.DeviceProfile;
import com.omoipassion.viton.pipeline.TryOnException;
import com.omoipassion.viton.pipeline.TryOnPipeline;
import com.omoipassion.viton.pipeline.TryOnResult;
import com.omoipassion.viton.util.AppExecutors;
import com.omoipassion.viton.util.BitmapLoader;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final String STATE_PERSON = "person_uri";
    private static final String STATE_GARMENT = "garment_uri";
    /** Inputs are decoded at most this size; the model only sees 256-512 px anyway. */
    private static final int MAX_INPUT_DIM = 1280;

    private ActivityMainBinding binding;
    private TryOnPipeline pipeline;

    private Uri personUri;
    private Uri garmentUri;
    private Bitmap person;
    private Bitmap garment;
    private boolean running;

    private final ActivityResultLauncher<Intent> captureLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), r -> {
                if (r.getResultCode() == RESULT_OK && r.getData() != null) {
                    String path = r.getData().getStringExtra(CaptureActivity.EXTRA_PHOTO_PATH);
                    if (path != null) {
                        load(Uri.fromFile(new File(path)), true);
                    }
                }
            });

    private final ActivityResultLauncher<String> pickPersonLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    load(uri, true);
                }
            });

    private final ActivityResultLauncher<String> pickGarmentLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    load(uri, false);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        pipeline = ((VitonApp) getApplication()).pipeline();
        showDeviceStatus(pipeline.profile());

        binding.capturePersonButton.setOnClickListener(
                v -> captureLauncher.launch(new Intent(this, CaptureActivity.class)));
        binding.pickPersonButton.setOnClickListener(v -> pickPersonLauncher.launch("image/*"));
        binding.pickGarmentButton.setOnClickListener(v -> pickGarmentLauncher.launch("image/*"));
        binding.tryOnButton.setOnClickListener(v -> runTryOn());

        if (savedInstanceState == null
                && getIntent().getBooleanExtra(WarpBenchmarkActivity.EXTRA_AUTORUN, false)) {
            startActivity(new Intent(this, WarpBenchmarkActivity.class)
                    .putExtra(WarpBenchmarkActivity.EXTRA_AUTORUN, true));
        }

        if (savedInstanceState != null) {
            Uri p = BundleCompat.getParcelable(savedInstanceState, STATE_PERSON, Uri.class);
            Uri g = BundleCompat.getParcelable(savedInstanceState, STATE_GARMENT, Uri.class);
            if (p != null) {
                load(p, true);
            }
            if (g != null) {
                load(g, false);
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == R.id.action_warp_benchmark) {
            startActivity(new Intent(this, WarpBenchmarkActivity.class));
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putParcelable(STATE_PERSON, personUri);
        outState.putParcelable(STATE_GARMENT, garmentUri);
    }

    private void load(Uri uri, boolean isPerson) {
        Context app = getApplicationContext();
        AppExecutors.io().execute(() -> {
            try {
                Bitmap bmp = BitmapLoader.decode(app, uri, MAX_INPUT_DIM);
                AppExecutors.runOnMain(() -> {
                    if (isDestroyed()) {
                        return;
                    }
                    if (isPerson) {
                        personUri = uri;
                        person = bmp;
                        binding.personImage.setImageBitmap(bmp);
                    } else {
                        garmentUri = uri;
                        garment = bmp;
                        binding.garmentImage.setImageBitmap(bmp);
                    }
                    updateTryOnButton();
                });
            } catch (IOException | SecurityException e) {
                // SecurityException: gallery URI grants don't survive process death.
                Log.w(TAG, "Load failed: " + uri, e);
                AppExecutors.runOnMain(() -> {
                    if (!isDestroyed()) {
                        Toast.makeText(this, R.string.error_load_image, Toast.LENGTH_SHORT).show();
                    }
                });
            }
        });
    }

    private void runTryOn() {
        if (person == null || garment == null || running) {
            return;
        }
        setRunning(true);
        binding.statusText.setText(R.string.processing);
        pipeline.run(person, garment, new TryOnPipeline.Callback() {
            @Override
            public void onSuccess(TryOnResult r) {
                if (isDestroyed()) {
                    return;
                }
                setRunning(false);
                binding.resultImage.setImageBitmap(r.image);
                binding.statusText.setText(getString(R.string.result_status,
                        r.modelName,
                        getString(r.usedGpu ? R.string.backend_gpu : R.string.backend_cpu),
                        r.poseMs, r.inferenceMs, r.totalMs));
            }

            @Override
            public void onError(TryOnException e) {
                if (isDestroyed()) {
                    return;
                }
                setRunning(false);
                binding.statusText.setText(e.reason.messageRes);
            }
        });
    }

    private void setRunning(boolean value) {
        running = value;
        binding.progress.setVisibility(value ? View.VISIBLE : View.GONE);
        updateTryOnButton();
    }

    private void updateTryOnButton() {
        binding.tryOnButton.setEnabled(person != null && garment != null && !running);
    }

    private void showDeviceStatus(DeviceProfile p) {
        binding.statusText.setText(getString(R.string.device_status,
                p.tier.name(), p.totalRamMb, getString(p.gpuUsable ? R.string.yes : R.string.no)));
    }
}
