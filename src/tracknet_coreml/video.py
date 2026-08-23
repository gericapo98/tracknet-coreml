"""Video reading and TrackNet input preprocessing.

Numerics deliberately mirror upstream TrackNetV3 (qaz812345/TrackNetV3, dataset.py):
PIL bicubic resizing at 512x288, BGR->RGB, median background sampled the same way,
and the same channel layout per bg_mode — so results are bit-comparable to the
reference implementation.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HEIGHT, WIDTH = 288, 512


def video_info(video_file: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_file}")
    info = {
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def compute_median(
    video_file: str | Path,
    max_sample_num: int = 1800,
    video_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Median background image, RGB float64 at full video resolution.

    Matches upstream __gen_median__: evenly sample up to max_sample_num frames
    (optionally restricted to [start_second, end_second]) and take the per-pixel median.
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_file}")
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if video_range is None:
        start_frame, end_frame = 0, video_len
    else:
        start_frame = max(0, video_range[0] * fps)
        end_frame = min(video_range[1] * fps, video_len)
    seg_len = end_frame - start_frame
    sample_step = seg_len // max_sample_num if seg_len > max_sample_num else 1

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pos = start_frame
    while pos < end_frame:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)  # BGR
        pos += 1
        for _ in range(sample_step - 1):  # cheap skip instead of seeking
            if pos >= end_frame or not cap.grab():
                pos = end_frame
                break
            pos += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames sampled from {video_file}")
    return np.median(frames, 0)[..., ::-1]  # BGR -> RGB


def prepare_median(median: np.ndarray, bg_mode: str) -> np.ndarray:
    """Median in the form __process__ expects: resized CHW uint8 for 'concat',
    full-resolution RGB float for the subtract modes."""
    if bg_mode == "concat":
        img = Image.fromarray(median.astype("uint8"))
        return np.moveaxis(np.array(img.resize(size=(WIDTH, HEIGHT))), -1, 0)
    return median


def process_window(frames_rgb: list[np.ndarray], bg_mode: str, median: np.ndarray | None) -> np.ndarray:
    """Stack a window of full-resolution RGB uint8 frames into the model input
    (C, 288, 512) float32 in [0, 1]. Mirrors upstream __process__ exactly."""
    out = []
    for frame in frames_rgb:
        img = Image.fromarray(frame)
        if bg_mode == "subtract":
            diff = Image.fromarray(np.sum(np.absolute(img - median), 2).astype("uint8"))
            out.append(np.array(diff.resize(size=(WIDTH, HEIGHT)))[None])
        elif bg_mode == "subtract_concat":
            diff = Image.fromarray(np.sum(np.absolute(img - median), 2).astype("uint8"))
            diff = np.array(diff.resize(size=(WIDTH, HEIGHT)))[None]
            rgb = np.moveaxis(np.array(img.resize(size=(WIDTH, HEIGHT))), -1, 0)
            out.append(np.concatenate((rgb, diff), axis=0))
        else:
            out.append(np.moveaxis(np.array(img.resize(size=(WIDTH, HEIGHT))), -1, 0))
    stacked = np.concatenate(out, axis=0)
    if bg_mode == "concat":
        stacked = np.concatenate((median, stacked), axis=0)
    return (stacked / 255.0).astype(np.float32)


def iter_windows(
    video_file: str | Path,
    seq_len: int,
    sliding_step: int,
    bg_mode: str,
    median: np.ndarray | None = None,
    pad_last: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (frame_indices (L,), input (C, 288, 512) float32) sliding windows.

    pad_last=True repeats the final frame to complete the last window (upstream's
    nonoverlap behavior); pad_last=False yields only complete windows (upstream's
    overlap/ensemble behavior).
    """
    if bg_mode:
        median = prepare_median(median, bg_mode)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_file}")

    frames: list[np.ndarray] = []
    indices: list[int] = []
    next_idx = 0
    success = True
    while success:
        while len(frames) < seq_len:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame[..., ::-1])  # BGR -> RGB
            indices.append(next_idx)
            next_idx += 1
        if not frames:
            break
        window_frames, window_indices = frames, indices
        if len(frames) < seq_len:
            if not pad_last:
                break
            pad = seq_len - len(frames)
            window_frames = frames + [frames[-1]] * pad
            window_indices = indices + [indices[-1]] * pad
        yield np.array(window_indices), process_window(window_frames, bg_mode, median)
        frames = frames[sliding_step:]
        indices = indices[sliding_step:]
    cap.release()
