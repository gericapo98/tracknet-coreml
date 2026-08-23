"""Heatmap and coordinate post-processing, mirroring upstream TrackNetV3 exactly.

Includes the sliding-window temporal ensemble (upstream predict.py's buffer logic,
reimplemented as a streaming generator), largest-blob localization, the inpaint
miss-mask heuristic, and CSV output in the upstream column order.
"""

import csv
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

HEIGHT, WIDTH = 288, 512
COOR_TH = 50 / math.sqrt(HEIGHT**2 + WIDTH**2)  # upstream DELTA_T * 50


def get_ensemble_weight(seq_len: int, eval_mode: str) -> np.ndarray:
    if eval_mode == "average":
        return np.ones(seq_len) / seq_len
    if eval_mode == "weight":
        weight = np.ones(seq_len)
        for i in range(math.ceil(seq_len / 2)):
            weight[i] = i + 1
            weight[seq_len - i - 1] = i + 1
        return weight / weight.sum()
    raise ValueError(f"invalid eval_mode: {eval_mode}")


def predict_location(heatmap: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of the largest-area blob in a binary uint8 heatmap;
    (0, 0, 0, 0) when the heatmap is empty."""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    best = max(rects, key=lambda r: r[2] * r[3])
    return best


def heatmap_to_coord(heatmap: np.ndarray, img_scaler: tuple[float, float]) -> tuple[int, int, int]:
    """One (H, W) float heatmap -> (x, y, visibility) in original video pixels.
    Threshold at 0.5, take the largest blob's bbox center, scale up — with the same
    two-step int truncation as upstream."""
    binary = ((heatmap > 0.5) * 255).astype(np.uint8)
    x, y, w, h = predict_location(binary)
    cx, cy = int(x + w / 2), int(y + h / 2)
    cx, cy = int(cx * img_scaler[0]), int(cy * img_scaler[1])
    vis = 0 if cx == 0 and cy == 0 else 1
    return cx, cy, vis


def ensemble_windows(
    window_preds: Iterator[np.ndarray],
    seq_len: int,
    eval_mode: str,
    num_windows: int | None = None,
) -> Iterator[np.ndarray]:
    """Streaming temporal ensemble over stride-1 sliding-window predictions.

    Takes stride-1 window predictions of shape (seq_len, ...) and yields one ensembled
    prediction of shape (...) per video frame (windows + seq_len - 1 of them). The end
    of the video is discovered from the iterator; num_windows, when given, is only a
    consistency check (container metadata can over-report frame counts).
    Frame f is the weighted combination of window w's position (f - w) over all
    windows containing f — upstream predict.py's anti-diagonal buffer, with its
    exact divisor conventions: early frames average over the windows seen so far,
    interior frames use the eval_mode weights, and tail frames divide by
    (seq_len - offset) regardless of how many windows actually contributed.
    """
    weight = get_ensemble_weight(seq_len, eval_mode)
    buffer: list[np.ndarray] = []  # last <= seq_len window predictions
    s = -1
    for s, pred in enumerate(window_preds):
        buffer.append(pred)
        if len(buffer) > seq_len:
            buffer.pop(0)
        # frame s: windows max(0, s-seq_len+1)..s, window w at position s-w
        contributions = [buffer[len(buffer) - 1 - k][k] for k in range(len(buffer))]
        if s < seq_len - 1:
            yield sum(contributions) / (s + 1)
        else:
            yield sum(w * c for w, c in zip(weight, contributions[::-1]))
    if s < 0:
        raise ValueError("no complete windows: video is shorter than seq_len")
    if num_windows is not None and s + 1 != num_windows:
        raise ValueError(f"expected {num_windows} windows, got {s + 1}")
    # tail frames s+f for f in 1..seq_len-1, averaged over remaining windows
    # (upstream divides by seq_len - f even when fewer windows exist)
    for f in range(1, seq_len):
        n = min(len(buffer), seq_len - f)
        contributions = [buffer[len(buffer) - 1 - k][k + f] for k in range(n)]
        yield sum(contributions) / (seq_len - f)


def generate_inpaint_mask(y: list[int], vis: list[int], th_h: float) -> list[int]:
    """Mark gap segments of the trajectory that InpaintNet should fill — gaps whose
    neighboring detections are below th_h pixels from the top (i.e. not the ball
    leaving the frame). Direct port of upstream generate_inpaint_mask."""
    y = np.array(y)
    vis_pred = np.array(vis)
    inpaint_mask = np.zeros_like(y)
    i = 0
    j = 0
    while j < len(vis_pred):
        while i < len(vis_pred) - 1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred) - 1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > th_h:
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i - 1] > th_h) and (j < len(vis_pred) and y[j] > th_h):
            inpaint_mask[i:j] = 1
        i = j
    return inpaint_mask.tolist()


def coord_to_pixel(c: np.ndarray, img_scaler: tuple[float, float]) -> tuple[int, int, int]:
    """Normalized (x, y) in [0,1] -> original video pixels + visibility, matching
    upstream's int truncation (c * 512 * w_scaler == c * video_width)."""
    cx, cy = int(c[0] * WIDTH * img_scaler[0]), int(c[1] * HEIGHT * img_scaler[1])
    vis = 0 if cx == 0 and cy == 0 else 1
    return cx, cy, vis


def write_csv(pred: dict, path: str | Path) -> None:
    """Frame,Visibility,X,Y — the upstream column order."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        for row in zip(pred["Frame"], pred["Visibility"], pred["X"], pred["Y"]):
            writer.writerow(row)
