"""Track a ball through a video and write per-frame coordinates to CSV.

    tracknet-track video.mp4 --tracknet TrackNet.mlpackage
    tracknet-track video.mp4 --tracknet TrackNet.mlpackage --inpaintnet InpaintNet.mlpackage
    tracknet-track video.mp4 --tracknet TrackNet_best.pt --device mps --fp16

Backend is picked by file type: .mlpackage runs via Core ML (Neural Engine by
default), .pt/.pth runs via PyTorch. Output CSV columns: Frame,Visibility,X,Y.
"""

import argparse
import time
from pathlib import Path

from tracknet_coreml.infer import load_backend, track_video
from tracknet_coreml.postprocess import write_csv
from tracknet_coreml.video import video_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="input video file")
    parser.add_argument("--tracknet", required=True, help="TrackNet .mlpackage or .pt")
    parser.add_argument("--inpaintnet", help="optional InpaintNet .mlpackage or .pt")
    parser.add_argument("--eval-mode", default="weight", choices=["nonoverlap", "average", "weight"],
                        help="nonoverlap: fast, one pass per frame; average/weight: stride-1 ensemble")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="PyTorch backend device")
    parser.add_argument("--fp16", action="store_true", help="PyTorch backend half precision")
    parser.add_argument("--compute-units", default="CPU_AND_NE",
                        choices=["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL"],
                        help="Core ML backend compute units")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sample-num", type=int, default=1800,
                        help="max frames sampled for the median background image")
    parser.add_argument("--video-range", type=lambda s: tuple(int(x) for x in s.split(",")),
                        help="start_second,end_second range for the median background image")
    parser.add_argument("-o", "--out", help="output CSV path (default: <video stem>_ball.csv)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    tracknet = load_backend(args.tracknet, device=args.device, fp16=args.fp16,
                            kind="tracknet", compute_units=args.compute_units)
    inpaintnet = None
    if args.inpaintnet:
        inpaintnet = load_backend(args.inpaintnet, device=args.device,
                                  kind="inpaintnet", compute_units=args.compute_units)

    start = time.perf_counter()
    pred = track_video(
        args.video, tracknet, inpaintnet=inpaintnet, eval_mode=args.eval_mode,
        batch_size=args.batch_size, max_sample_num=args.max_sample_num,
        video_range=args.video_range, progress=not args.quiet,
    )
    elapsed = time.perf_counter() - start

    out_path = Path(args.out) if args.out else Path(args.video).with_name(f"{Path(args.video).stem}_ball.csv")
    write_csv(pred, out_path)
    if not args.quiet:
        info = video_info(args.video)
        print(f"wrote {out_path} — {len(pred['Frame'])} frames in {elapsed:.1f}s "
              f"({len(pred['Frame']) / elapsed:.1f} frames/s, video is {info['fps']:.0f} fps)")


if __name__ == "__main__":
    main()
