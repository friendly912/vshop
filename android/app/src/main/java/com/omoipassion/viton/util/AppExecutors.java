package com.omoipassion.viton.util;

import android.os.Handler;
import android.os.Looper;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/** Process-wide executors. */
public final class AppExecutors {

    /**
     * Single thread for everything that touches the interpreter. The LiteRT GPU delegate is
     * bound to the thread that created it, so create, run and close must all happen here.
     */
    private static final ExecutorService INFERENCE =
            Executors.newSingleThreadExecutor(named("viton-inference"));

    /** Image decoding and file I/O. */
    private static final ExecutorService IO =
            Executors.newFixedThreadPool(2, named("viton-io"));

    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    private AppExecutors() {
    }

    public static ExecutorService inference() {
        return INFERENCE;
    }

    public static ExecutorService io() {
        return IO;
    }

    public static void runOnMain(Runnable r) {
        MAIN.post(r);
    }

    private static ThreadFactory named(String name) {
        return r -> {
            Thread t = new Thread(r, name);
            t.setDaemon(true);
            return t;
        };
    }
}
