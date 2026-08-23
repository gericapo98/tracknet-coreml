# tracknet-coreml

Run [TrackNetV3](https://github.com/qaz812345/TrackNetV3) shuttlecock/ball tracking
on Apple Silicon — a Core ML converter, on-device inference, and benchmarks.

TrackNetV3 (and the TrackNet family generally) ships as PyTorch/CUDA code; on a Mac it
falls back to CPU and full-match videos take tens of minutes to hours. As of August 2026
no Core ML or MLX port of TrackNetV3 exists. This project closes that gap in stages:

1. **MPS baseline** (done) — load any TrackNetV3-family checkpoint and benchmark it on
   CPU vs. the Apple GPU (`tracknet-bench`), establishing the number the port must beat.
2. **Core ML conversion** — export the TrackNet U-Net (and the InpaintNet rectifier) to
   fp16 `mlprogram` packages targeting the Neural Engine via coremltools.
3. **Inference API** — frames in → ball coordinates out, with the heatmap post-processing
   (threshold, centroid, optional rectification/ensembling) reimplemented host-side, plus
   parity tests against the PyTorch reference.

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
```

```python
from tracknet_coreml import load_tracknet

model, config = load_tracknet("TrackNet_best.pt")   # model.eval(), TrackNetConfig
```

## Attribution

Model definitions are vendored unmodified from
[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) (MIT). This repo is
also MIT — see [LICENSE](LICENSE).
