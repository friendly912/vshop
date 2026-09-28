package com.omoipassion.viton.pipeline;

import androidx.annotation.StringRes;

import com.omoipassion.viton.R;

public final class TryOnException extends Exception {

    public enum Reason {
        MODEL_MISSING(R.string.error_model_missing),
        NO_PERSON(R.string.error_no_person),
        TORSO_NOT_VISIBLE(R.string.error_torso_not_visible),
        OUT_OF_MEMORY(R.string.error_out_of_memory),
        INTERNAL(R.string.error_internal);

        @StringRes
        public final int messageRes;

        Reason(@StringRes int messageRes) {
            this.messageRes = messageRes;
        }
    }

    public final Reason reason;

    public TryOnException(Reason reason) {
        super(reason.name());
        this.reason = reason;
    }

    public TryOnException(Reason reason, Throwable cause) {
        super(reason.name(), cause);
        this.reason = reason;
    }
}
