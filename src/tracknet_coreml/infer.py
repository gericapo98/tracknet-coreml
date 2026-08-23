"""End-to-end tracking: video in -> per-frame ball coordinates out.

Backends run the same pipeline: Core ML mlpackages (from tracknet-convert) or the
original PyTorch checkpoints (cpu/mps). Post-processing matches upstream TrackNetV3
predict.py: nonoverlap or stride-1 temporal-ensemble modes, plus optional InpaintNet
trajectory rectification.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from tracknet_coreml.postprocess import (
    COOR_TH,
    HEIGHT,
    WIDTH,
    coord_to_pixel,
    ensemble_windows,
    generate_inpaint_mask,
    heatmap_to_coord,
)
from tracknet_coreml.video import compute_median, iter_windows, video_info

INPAINT_SEQ_LEN = 16


# --- backends ---------------------------------------------------------------

class CoreMLTrackNet:
    def __init__(self, mlpackage_path: str | Path, compute_units: str = "CPU_AND_NE"):
        import coremltools as ct

        self.model = ct.models.MLModel(
            str(mlpackage_path), compute_units=getattr(ct.ComputeUnit, compute_units)
        )
        meta = self.model.user_defined_metadata
        spec = self.model.get_spec().description
        self.input_name = spec.input[0].name
        self.output_name = spec.output[0].name
        in_dim, out_dim = None, None
        shape = spec.input[0].type.multiArrayType.shape
        if len(shape) == 4:
            in_dim = int(shape[1])
        self.seq_len = int(meta["seq_len"])
        self.bg_mode = meta.get("bg_mode", "")
        self.in_dim = in_dim

    def predict(self, batch: np.ndarray) -> np.ndarray:
        out = [
            self.model.predict({self.input_name: batch[i : i + 1]})[self.output_name]
            for i in range(batch.shape[0])
        ]
        return np.concatenate(out, axis=0)


class CoreMLInpaintNet:
    def __init__(self, mlpackage_path: str | Path, compute_units: str = "CPU_AND_NE"):
        import coremltools as ct

        self.model = ct.models.MLModel(
            str(mlpackage_path), compute_units=getattr(ct.ComputeUnit, compute_units)
        )
        spec = self.model.get_spec().description
        self.input_names = [inp.name for inp in spec.input]
        self.output_name = spec.output[0].name
        self.seq_len = int(self.model.user_defined_metadata.get("seq_len", INPAINT_SEQ_LEN))

    def predict(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = [
            self.model.predict(
                {self.input_names[0]: coords[i : i + 1], self.input_names[1]: mask[i : i + 1]}
            )[self.output_name]
            for i in range(coords.shape[0])
        ]
        return np.concatenate(out, axis=0)


class TorchTrackNet:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu", fp16: bool = False):
        import torch

        from tracknet_coreml.checkpoint import load_tracknet

        self.torch = torch
        model, config = load_tracknet(ckpt_path)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model = model.to(device=device, dtype=self.dtype)
        self.device = device
        self.seq_len = config.seq_len
        self.bg_mode = config.bg_mode
        self.in_dim = config.in_dim

    def predict(self, batch: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(batch).to(device=self.device, dtype=self.dtype)
        with self.torch.no_grad():
            y = self.model(x)
        return y.float().cpu().numpy()


class TorchInpaintNet:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu"):
        import torch

        from tracknet_coreml.checkpoint import load_inpaintnet

        self.torch = torch
        self.model = load_inpaintnet(ckpt_path).to(device)
        self.device = device
        self.seq_len = INPAINT_SEQ_LEN

    def predict(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        c = self.torch.from_numpy(coords).float().to(self.device)
        m = self.torch.from_numpy(mask).float().to(self.device)
        with self.torch.no_grad():
            out = self.model(c, m)
        return out.float().cpu().numpy()


def load_backend(path: str | Path, device: str = "cpu", fp16: bool = False,
                 kind: str = "tracknet", compute_units: str = "CPU_AND_NE"):
    """Pick a backend by file type: .mlpackage -> Core ML, .pt/.pth -> PyTorch."""
    path = Path(path)
    if path.suffix == ".mlpackage":
        cls = CoreMLTrackNet if kind == "tracknet" else CoreMLInpaintNet
        return cls(path, compute_units=compute_units)
    if kind == "tracknet":
        return TorchTrackNet(path, device=device, fp16=fp16)
    return TorchInpaintNet(path, device=device)


# --- pipeline ---------------------------------------------------------------

def _batched(iterator: Iterator, batch_size: int) -> Iterator[list]:
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def track_video(
    video_file: str | Path,
    tracknet,
    inpaintnet=None,
    eval_mode: str = "weight",
    batch_size: int = 8,
    max_sample_num: int = 1800,
    video_range: tuple[int, int] | None = None,
    median: np.ndarray | None = None,
    progress: bool = False,
) -> dict:
    """Track the ball through a video. Returns {'Frame': [...], 'X': [...],
    'Y': [...], 'Visibility': [...]} in original video pixel coordinates."""
    info = video_info(video_file)
    img_scaler = (info["width"] / WIDTH, info["height"] / HEIGHT)
    seq_len = tracknet.seq_len
    if tracknet.bg_mode and median is None:
        if progress:
            print("computing median background...")
        median = compute_median(video_file, max_sample_num=max_sample_num, video_range=video_range)

    pred = {"Frame": [], "X": [], "Y": [], "Visibility": []}

    def emit(frame_idx: int, cx: int, cy: int, vis: int) -> None:
        pred["Frame"].append(frame_idx)
        pred["X"].append(cx)
        pred["Y"].append(cy)
        pred["Visibility"].append(vis)

    if eval_mode == "nonoverlap":
        windows = iter_windows(video_file, seq_len, seq_len, tracknet.bg_mode, median, pad_last=True)
        prev_idx = -1
        for batch in _batched(windows, batch_size):
            indices, inputs = zip(*batch)
            y = tracknet.predict(np.stack(inputs))
            for window_indices, heatmaps in zip(indices, y):
                for pos in range(seq_len):
                    f_i = int(window_indices[pos])
                    if f_i == prev_idx:
                        break  # padding repeats at the end of the last window
                    emit(f_i, *heatmap_to_coord(heatmaps[pos], img_scaler))
                    prev_idx = f_i
            if progress:
                print(f"\rtracknet: {prev_idx + 1}/{info['num_frames']} frames", end="", flush=True)
    else:
        num_windows = info["num_frames"] - seq_len + 1
        if num_windows < 1:
            raise ValueError(
                f"video has {info['num_frames']} frames, fewer than seq_len={seq_len}; "
                "use eval_mode='nonoverlap'"
            )
        windows = iter_windows(video_file, seq_len, 1, tracknet.bg_mode, median, pad_last=False)

        def window_preds() -> Iterator[np.ndarray]:
            done = 0
            for batch in _batched(windows, batch_size):
                _, inputs = zip(*batch)
                y = tracknet.predict(np.stack(inputs))
                done += len(batch)
                if progress:
                    print(f"\rtracknet: {done}/{num_windows} windows", end="", flush=True)
                yield from y

        for f_i, heatmap in enumerate(ensemble_windows(window_preds(), seq_len, eval_mode, num_windows)):
            emit(f_i, *heatmap_to_coord(heatmap, img_scaler))
    if progress:
        print()

    if inpaintnet is not None:
        pred = rectify_trajectory(
            pred, inpaintnet, eval_mode=eval_mode, batch_size=batch_size,
            video_size=(info["width"], info["height"]),
        )
    return pred


def rectify_trajectory(
    pred: dict,
    inpaintnet,
    video_size: tuple[int, int],
    eval_mode: str = "weight",
    batch_size: int = 8,
) -> dict:
    """InpaintNet rectification of a TrackNet trajectory, matching upstream:
    coordinates normalized by video size, miss-mask from the height heuristic,
    inpainted values blended only where the mask is set, low-coordinate results
    zeroed, and the same nonoverlap/ensemble window schemes as the heatmap stage."""
    w, h = video_size
    seq_len = inpaintnet.seq_len
    img_scaler = (w / WIDTH, h / HEIGHT)
    n = len(pred["Frame"])
    mask_list = generate_inpaint_mask(pred["Y"], pred["Visibility"], th_h=h * 0.05)
    coords = np.stack(
        [np.array(pred["X"], dtype=np.float32) / w, np.array(pred["Y"], dtype=np.float32) / h], axis=-1
    )
    mask = np.array(mask_list, dtype=np.float32)[:, None]

    def blend(idx_windows: np.ndarray) -> np.ndarray:
        """Run InpaintNet on coordinate windows given by index array (N, L)."""
        c = coords[idx_windows]  # (N, L, 2)
        m = mask[idx_windows]  # (N, L, 1)
        out = inpaintnet.predict(c, m)
        out = out * m + c * (1 - m)
        low = (out[:, :, 0] < COOR_TH) & (out[:, :, 1] < COOR_TH)
        out[low] = 0.0
        return out

    result = {"Frame": [], "X": [], "Y": [], "Visibility": []}

    def emit(frame_idx: int, c: np.ndarray) -> None:
        cx, cy, vis = coord_to_pixel(c, img_scaler)
        result["Frame"].append(frame_idx)
        result["X"].append(cx)
        result["Y"].append(cy)
        result["Visibility"].append(vis)

    if eval_mode == "nonoverlap":
        starts = list(range(0, n, seq_len))
        idx_windows = np.array(
            [[min(s + f, n - 1) for f in range(seq_len)] for s in starts]
        )
        prev_idx = -1
        for i in range(0, len(idx_windows), batch_size):
            chunk = idx_windows[i : i + batch_size]
            out = blend(chunk)
            for window_idx, window_out in zip(chunk, out):
                for pos in range(seq_len):
                    f_i = int(window_idx[pos])
                    if f_i == prev_idx:
                        break
                    emit(f_i, window_out[pos])
                    prev_idx = f_i
    else:
        num_windows = n - seq_len + 1
        if num_windows < 1:
            raise ValueError(f"trajectory has {n} frames, fewer than seq_len={seq_len}")
        idx_windows = np.array([np.arange(s, s + seq_len) for s in range(num_windows)])

        def window_preds() -> Iterator[np.ndarray]:
            for i in range(0, num_windows, batch_size):
                yield from blend(idx_windows[i : i + batch_size])

        for f_i, c in enumerate(ensemble_windows(window_preds(), seq_len, eval_mode, num_windows)):
            c = np.where((c[0] < COOR_TH) & (c[1] < COOR_TH), 0.0, c)  # re-threshold after ensembling
            emit(f_i, c)

    return result
