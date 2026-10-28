# Desktop interaction acceptance

Confirmed on the host: the CUDA-backed dashboard opens and stays visible after the GTK window-lifetime fix. Offscreen rendering tests pass. A display-independent interaction smoke test now passes on CPU C++20/C++23 and ASan/UBSan, and also passes with the CUDA backend on RTX 3080. Saved host artifacts are in examples/gpu-gui-actions. It invokes the shared application key/mouse handlers, checking control, edits, guards, reset, forecasting, presets, filtering, endpoint edits and save. Native OS event delivery and window-manager behavior are not covered. The following desktop checks remain separate.

Run `build/cuda/hzl-gui --backend cuda` on the desktop host.

1. Press N: timestamp advances one control interval, observations change, endpoint forecast is labeled, baseline and error metrics update. Press F while paused: forecast changes to the preview horizon but process time stays fixed.
2. Press Space: automatic steps continue until the endpoint, then pause with an endpoint message. Press R: time returns to zero and fields reset. Endpoint stopping means the experiment ended, not that physical diffusion stops.
3. Press A to choose manual mode; edit the temperature panel while paused. Use [ and ] to adjust the Kelvin brush. Step and verify the edited map is used. Return to automatic mode and verify bounds are maintained.
4. At time zero, paint concentration: the process resets and a new reachable target is generated. Paint the target panel: target mismatch and mass incompatibility may remain, without a fabricated success indication.
5. Press 1–4 for initial presets, T for temperature presets, B for filter bypass/on, G for a new target seed, and E for an extended endpoint. Each reset should visibly preserve labels and chronological semantics.
6. Press S: inspect saved fields, cycle diagnostics, metadata and dashboard image. Close with Q/Esc and the window manager close button; each should exit cleanly and print the reason.

The application is synchronous stepped mode. Headless controller deadline results exclude GUI drawing, event handling and presentation; actual desktop responsiveness is part of this acceptance check.
