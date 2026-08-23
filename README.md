# tracknet-coreml

Run [TrackNetV3](https://github.com/qaz812345/TrackNetV3) shuttlecock/ball tracking
on Apple Silicon — a Core ML converter, on-device inference, and benchmarks.

TrackNetV3 (and the TrackNet family generally) ships as PyTorch/CUDA code; on a Mac it
falls back to CPU and full-match videos take tens of minutes to hours. As of August 2026
no Core ML or MLX port of TrackNetV3 exists. This project closes that gap with four tools:

1. **Benchmarking** — `tracknet-bench` runs any TrackNetV3-family checkpoint on CPU or
   the Apple GPU (MPS) so you can measure throughput on your own machine.
2. **Core ML conversion** — `tracknet-convert` exports the TrackNet U-Net and the
   InpaintNet rectifier to fp16 `mlprogram` packages targeting the Neural Engine, with a
   built-in parity check against the fp32 PyTorch reference.
3. **Inference** — `tracknet-track` runs a video through either backend (Core ML or
   PyTorch) and writes per-frame ball coordinates, with upstream's post-processing
   (threshold/centroid, nonoverlap or temporal-ensemble modes, optional InpaintNet
   rectification) reimplemented host-side.
4. **Overlay** — `tracknet-overlay` renders the tracked trajectory onto the video so you
   can watch the result and judge the tracking yourself.

Works with any checkpoint in the upstream format (`{'param_dict': ..., 'model': ...}`)
or a bare state dict — sequence length and background mode are read from the checkpoint,
so tennis/volleyball/other-sport TrackNet weights load the same way as badminton ones.

## Usage

```bash
pip install -e .

# benchmark a checkpoint on CPU and MPS, fp32 and fp16
tracknet-bench --ckpt path/to/TrackNet_best.pt

# no checkpoint handy? benchmark the architecture with random weights
tracknet-bench --seq-len 8 --bg-mode concat

# convert to Core ML (needs the [convert] extra)
tracknet-convert --ckpt TrackNet_best.pt -o TrackNet.mlpackage
tracknet-convert --ckpt InpaintNet_best.pt --kind inpaintnet -o InpaintNet.mlpackage

# track a video on the Neural Engine, then watch the result
tracknet-track rally.mp4 --tracknet TrackNet.mlpackage --inpaintnet InpaintNet.mlpackage
tracknet-overlay rally.mp4 rally_ball.csv          # writes rally_tracked.mp4

# or run the original checkpoint via PyTorch (cpu/mps) — same pipeline
tracknet-track rally.mp4 --tracknet TrackNet_best.pt --device mps --fp16
```

`tracknet-track` defaults to upstream's max-accuracy stride-1 temporal ensemble
(`--eval-mode weight`); use `--eval-mode nonoverlap` for the fast single-pass mode.

```python
from tracknet_coreml import load_tracknet
from tracknet_coreml.infer import load_backend, track_video

model, config = load_tracknet("TrackNet_best.pt")   # torch model.eval(), TrackNetConfig

tracknet = load_backend("TrackNet.mlpackage")        # or a .pt checkpoint
pred = track_video("rally.mp4", tracknet)            # {'Frame', 'X', 'Y', 'Visibility'}
```

Conversion note: torch versions newer than coremltools' tested range trigger an
"untested version" warning but generally convert fine; pin `torch<=2.7` if you hit
converter issues.

## Attribution

Model definitions are vendored unmodified from
[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) (MIT). This repo is
also MIT — see [LICENSE](LICENSE).
