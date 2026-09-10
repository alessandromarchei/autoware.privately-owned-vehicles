# VisionPilot Remote Host — production/debug/occupancy UI

This package keeps the existing VPNT v2 / FlatBuffers transport unchanged and replaces only the host-side application/visualization layer.

## Files

- `host.py` — Linux host application. Frame/result synchronization is kept; adds debug controls, runtime statistics and mouse forwarding to the embedded occupancy panel.
- `visualization.py` — production camera view, C++-style engineering debug camera view, debug-level side panels and occupancy bridge.
- `occupancy_view.py` — Python port of the heuristic C++ 3D occupancy renderer, including orbit/pan/zoom.
- `wire.py` — unchanged copy of the existing wire implementation for completeness.

Keep the existing generated `visionpilot/wire/*` FlatBuffers Python bindings next to/in the Python environment exactly as before.

## Debug levels

`--debug-level 0`: production VisionPilot camera HUD + 3D occupancy, no diagnostic side panel.

`--debug-level 1`: adds compact runtime, network, fusion and planner telemetry.

`--debug-level 2`: adds per-model AutoDrive / AutoSteer / AutoSpeed detail and useful statistics.

`--debug-level 3`: shows the complete values actually transported from V4M to the host: AutoDrive scalars, all AutoSteer `xp` / `h_vector` values, post-NMS AutoSpeed detections, CIPO/lateral fusion and planner state.

The current protocol does not transport the pre-NMS AutoSpeed tensor or separate per-model `ad_ms/as_ms/asp_ms` timings. Those require a FlatBuffers/V4M-side protocol extension if needed later.

## Typical commands

Clean production dashboard, matching the C++ application structure:

```bash
python host.py \
  --video input.mp4 \
  --realtime \
  --speed 15.0 \
  --speed-limit 22.22 \
  --homography H.yaml \
  --debug-level 0
```

Detailed model debugging while preserving the production camera view:

```bash
python host.py \
  --frames /path/to/frames \
  --speed-file vehicle_speed.csv \
  --homography H.yaml \
  --debug-level 2 \
  --display-width 2560
```

Maximum GUI debug without flooding stdout:

```bash
python host.py \
  --frames /path/to/frames \
  --homography H.yaml \
  --debug-level 3 \
  --console-level 0 \
  --display-width 2560
```

C++ engineering/debug camera overlay + 3D occupancy + model detail:

```bash
python host.py \
  --video input.mp4 \
  --homography H.yaml \
  --camera-view debug \
  --debug-level 2 \
  --wheel-dir /path/to/VisionPilot/development_releases/0.9/images
```

For production warning icons, pass the same directory used by VisionPilot C++:

```bash
--icons-dir /path/to/VisionPilot/assets/icons
```

## Runtime controls

- `0`, `1`, `2`, `3`: change GUI debug level live.
- `V`: switch production/debug camera view live.
- `O`: toggle 3D occupancy.
- Left-drag inside occupancy: orbit camera.
- Right/middle-drag inside occupancy: pan.
- Mouse wheel or `+` / `-`: zoom.
- `R`: reset occupancy camera.
- `Q` or `Esc`: quit.

## Notes

The visualizer renders the camera view in the same 1024x512 inference plane used by the C++ application before assembling the host dashboard. `--display-width 0` keeps the canonical dashboard dimensions without final scaling; for a 2560-wide monitor, `--display-width 2560` is recommended for debug level 2/3 readability.

If no `--homography` is supplied, the renderer uses the same hard-coded warped-pixel-to-world fallback matrix already used by the VisionPilot C++ visualization. For accurate detection placement in the 3D panel with a plain-resized input image, use the real `H_resized`-equivalent calibration whenever available.
