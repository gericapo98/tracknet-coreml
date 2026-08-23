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

## Baseline results

TrackNetV3 badminton checkpoint (`seq_len=8`, `bg_mode='concat'`, 27→8 channels at
288×512), Apple M4 Pro, torch 2.12.1, batch 1 (the GPU is already saturated at batch 1;
batches 4/8 measure the same):

| device | precision | ms/pass | output frames/s |
|--------|-----------|--------:|----------------:|
| CPU    | fp32      |   177.5 |            45.1 |
| MPS    | fp32      |    41.3 |           193.9 |
| MPS    | fp16      |    36.2 |           221.3 |

fp16 on MPS is numerically safe for this model: against the fp32 CPU reference the
max heatmap deviation is 6e-4 and the argmax (ball position) is identical on every
output channel. Frames/s counts non-overlapping windows; upstream's optional 8×
overlap ensembling divides these numbers by 8.

## Attribution

Model definitions are vendored unmodified from
[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) (MIT). This repo is
also MIT — see [LICENSE](LICENSE).
