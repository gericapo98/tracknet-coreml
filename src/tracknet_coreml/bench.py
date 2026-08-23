"""Benchmark TrackNet inference across devices and precisions.

This is the baseline harness the Core ML port is measured against:

    tracknet-bench --ckpt ckpts/TrackNet_best.pt                 # cpu + mps, fp32 + fp16
    tracknet-bench --device mps --precision fp16 --batch 4       # one configuration
    tracknet-bench                                               # random weights, default config

Reported frames/s counts *output* frames (one pass of a seq_len-in/seq_len-out
model yields seq_len tracked frames), which is the number comparable to video fps.
"""

import argparse
import time

import torch

from tracknet_coreml.checkpoint import TrackNetConfig, _dims_for, load_tracknet
from tracknet_coreml.model import TrackNet


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def run_once(
    model: torch.nn.Module,
    config: TrackNetConfig,
    device: str,
    precision: str,
    batch: int,
    iters: int,
    warmup: int,
) -> dict:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = model.to(device=device, dtype=dtype)
    x = torch.rand(batch, config.in_dim, 288, 512, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        _sync(device)
        elapsed = time.perf_counter() - start

    ms_per_pass = elapsed / iters * 1000 / batch
    frames_per_s = 1000 / ms_per_pass * config.out_dim
    return {
        "device": device,
        "precision": precision,
        "batch": batch,
        "ms_per_pass": ms_per_pass,
        "frames_per_s": frames_per_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", help="TrackNet checkpoint (.pt); omit to benchmark random weights")
    parser.add_argument("--device", choices=["cpu", "mps"], help="default: cpu and mps (if available)")
    parser.add_argument("--precision", choices=["fp32", "fp16"], help="default: fp32 and fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=8, help="without --ckpt: sequence length")
    parser.add_argument("--bg-mode", default="concat", choices=["", "subtract", "subtract_concat", "concat"],
                        help="without --ckpt: background mode")
    args = parser.parse_args()

    if args.ckpt:
        model, config = load_tracknet(args.ckpt)
        source = args.ckpt
    else:
        in_dim, out_dim = _dims_for(args.seq_len, args.bg_mode)
        config = TrackNetConfig(args.seq_len, args.bg_mode, in_dim, out_dim)
        model = TrackNet(in_dim=in_dim, out_dim=out_dim).eval()
        source = "random weights"

    devices = [args.device] if args.device else ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    precisions = [args.precision] if args.precision else ["fp32", "fp16"]

    print(f"TrackNet {config.in_dim}ch->{config.out_dim}ch @ 288x512  ({source}, "
          f"seq_len={config.seq_len}, bg_mode={config.bg_mode!r}, batch={args.batch})")
    print(f"{'device':<6} {'prec':<5} {'ms/pass':>9} {'frames/s':>9}")
    for device in devices:
        for precision in precisions:
            if device == "cpu" and precision == "fp16":
                continue  # fp16 conv on CPU is unsupported or pathologically slow
            r = run_once(model, config, device, precision, args.batch, args.iters, args.warmup)
            print(f"{r['device']:<6} {r['precision']:<5} {r['ms_per_pass']:>9.1f} {r['frames_per_s']:>9.1f}")


if __name__ == "__main__":
    main()
