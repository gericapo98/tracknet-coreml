# tracknet-coreml

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)

Run [TrackNetV3](https://github.com/qaz812345/TrackNetV3) shuttlecock/ball tracking
on Apple Silicon. TrackNet ships as PyTorch/CUDA code and falls back to CPU on a Mac;
this project converts it to Core ML so it runs on the Neural Engine.

## Quickstart

```bash
pip install -e '.[convert]'

# convert a TrackNetV3 checkpoint to Core ML
tracknet-convert --ckpt TrackNet_best.pt -o TrackNet.mlpackage
tracknet-convert --ckpt InpaintNet_best.pt --kind inpaintnet -o InpaintNet.mlpackage

# track a video on the Neural Engine, then watch the result
tracknet-track rally.mp4 --tracknet TrackNet.mlpackage --inpaintnet InpaintNet.mlpackage
tracknet-overlay rally.mp4 rally_ball.csv    # writes rally_tracked.mp4
```

```python
from tracknet_coreml.infer import load_backend, track_video

pred = track_video("rally.mp4", load_backend("TrackNet.mlpackage"))
# {'Frame': [...], 'X': [...], 'Y': [...], 'Visibility': [...]}
```

## Tools

| command | what it does |
|---|---|
| `tracknet-convert` | checkpoint → fp16 Core ML `mlpackage` (Neural Engine), with parity check |
| `tracknet-track` | video → per-frame ball coordinates CSV (Core ML or PyTorch backend) |
| `tracknet-overlay` | video + CSV → mp4 with the tracked trajectory drawn on it |
| `tracknet-bench` | measure model throughput on your machine (CPU / MPS) |

Post-processing matches upstream `predict.py`: median background, nonoverlap or
stride-1 temporal-ensemble modes, and optional InpaintNet rectification. Any
checkpoint in the upstream format works — `seq_len` and `bg_mode` are read from the
file, so weights for other sports load the same way.

## Attribution

Model definitions are vendored unmodified from
[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) (MIT). This project
is MIT — see [LICENSE](LICENSE).
