#!/usr/bin/env bash
# Install the debug APK, run the warp benchmark without interaction, save the report.
# Works with any adb target: emulator, USB phone, or a device farm with remote adb.
#
# Usage: scripts/run_warp_bench.sh [-s SERIAL]
set -euo pipefail

cd "$(dirname "$0")/.."
ADB=(adb)
if [[ "${1:-}" == "-s" ]]; then
    ADB+=(-s "$2")
fi

APK=app/build/outputs/apk/debug/app-debug.apk
PKG=com.omoipassion.viton
TIMEOUT_S=300

[[ -f "$APK" ]] || ./gradlew assembleDebug --console=plain -q

"${ADB[@]}" install -r -t "$APK" >/dev/null
"${ADB[@]}" shell am force-stop "$PKG"
"${ADB[@]}" logcat -c
"${ADB[@]}" shell am start -n "$PKG/.ui.MainActivity" --ez warp_autorun true >/dev/null

model=$("${ADB[@]}" shell getprop ro.product.model | tr -d '\r' | tr ' /' '__')
out="../ml/warp_spike/results/device_${model}_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$out")"

echo "Running on $model (up to ${TIMEOUT_S}s)..."
for ((i = 0; i < TIMEOUT_S; i += 5)); do
    if "${ADB[@]}" logcat -d -s WarpBench:I | grep -q WARP_BENCH_DONE; then
        break
    fi
    sleep 5
done

"${ADB[@]}" logcat -d -v raw -s WarpBench:I AndroidRuntime:E LiteRtRunner:W | grep -v '^---' > "$out"
cat "$out"
grep -q WARP_BENCH_DONE "$out" || { echo "Benchmark did not finish (see $out)" >&2; exit 1; }
echo "Saved $out"
