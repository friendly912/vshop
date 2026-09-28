package com.omoipassion.viton.util;

import android.content.Context;

import java.io.IOException;
import java.io.InputStream;

public final class Assets {

    private Assets() {
    }

    public static boolean exists(Context context, String path) {
        try (InputStream ignored = context.getAssets().open(path)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }
}
