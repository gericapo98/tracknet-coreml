"""Render tracking results onto a video so you can watch how the ball is tracked.

    tracknet-overlay video.mp4 video_ball.csv -o tracked.mp4

Draws the current detection as a marked point with a fading trail of recent
positions; frames where the ball was not detected are left unmarked.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_csv(path: str | Path) -> dict:
    pred = {"Frame": [], "X": [], "Y": [], "Visibility": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pred["Frame"].append(int(row["Frame"]))
            pred["X"].append(int(row["X"]))
            pred["Y"].append(int(row["Y"]))
            pred["Visibility"].append(int(row["Visibility"]))
    return pred


def render_overlay(
    video_file: str | Path,
    pred: dict,
    out_file: str | Path,
    traj_len: int = 8,
) -> Path:
    """Write a copy of the video with the tracked trajectory drawn on it."""
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_file = Path(out_file)
    writer = None
    for fourcc in ("avc1", "mp4v"):  # H.264 when the OS backend provides it
        writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError("could not open a video writer for .mp4 output")

    by_frame = {
        f: (x, y)
        for f, x, y, v in zip(pred["Frame"], pred["X"], pred["Y"], pred["Visibility"])
        if v == 1
    }
    scale = max(w / 1280, 1.0)  # marker sizes tuned for 720p, scaled for larger videos
    trail: list[tuple[int, int] | None] = []

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        trail.append(by_frame.get(frame_idx))
        if len(trail) > traj_len:
            trail.pop(0)

        # fading trail, oldest first
        for age, point in enumerate(trail[:-1]):
            if point is None:
                continue
            strength = (age + 1) / len(trail)
            radius = max(1, int((2 + 3 * strength) * scale))
            color = (0, int(150 + 105 * strength), int(255 * strength))  # BGR: green->yellow
            cv2.circle(frame, point, radius, color, -1, lineType=cv2.LINE_AA)
        # current detection: filled dot + ring
        if trail and trail[-1] is not None:
            point = trail[-1]
            cv2.circle(frame, point, max(2, int(5 * scale)), (0, 220, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, point, max(4, int(9 * scale)), (0, 0, 255), max(1, int(2 * scale)),
                       lineType=cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="input video file")
    parser.add_argument("csv", help="tracking CSV (Frame,Visibility,X,Y) from tracknet-track")
    parser.add_argument("-o", "--out", help="output video path (default: <video stem>_tracked.mp4)")
    parser.add_argument("--traj-len", type=int, default=8, help="trail length in frames")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(args.video).with_name(f"{Path(args.video).stem}_tracked.mp4")
    render_overlay(args.video, read_csv(args.csv), out, traj_len=args.traj_len)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
