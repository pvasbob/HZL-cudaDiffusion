# GPU-backed GUI action smoke test — 2026-10-26

Executed on the RTX 3080 host:

```sh
build/cuda/hzl-gui --backend cuda --interaction-smoke --output gpu-reports/gui-actions
```

The test reported PASS, covering shared GUI key/mouse actions, chronological steps, preview isolation, editing guards, manual control, presets, filtering, reset, endpoint and save. Desktop event delivery is explicitly not exercised by this test.

The saved artifacts are preserved here. Metadata confirms backend=cuda and NVIDIA GeForce RTX 3080. The final scenario has seed 43, endpoint 15s, gradient temperature preset, deposits concentration preset, and radius-2 filtering, as expected after the tested actions. fields.csv contains all 3185 cells with finite fields, nonnegative underlying/reconstructed concentration, and temperatures in [700,1100] K. dashboard.png, fields.csv, metadata.txt and cycles.csv exist. The final cycle CSV has only its header because the smoke scenario ends after reset/target regeneration; earlier step assertions are established by the passing test, not by final saved cycle rows.

CPU C++20/C++23 and host sanitizer action tests previously passed. Native window startup was separately confirmed after the GTK lifecycle fix. OS event delivery, window-manager close behavior and live presentation latency are still separate manual checks.
