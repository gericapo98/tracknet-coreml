"""Load TrackNetV3-family checkpoints into vendored model definitions.

Supports the upstream checkpoint layout ({'param_dict': {...}, 'model': state_dict})
and falls back to shape inference from the state dict alone, so checkpoints from
forks that dropped param_dict still load.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from tracknet_coreml.model import InpaintNet, TrackNet


@dataclass(frozen=True)
class TrackNetConfig:
    seq_len: int
    bg_mode: str
    in_dim: int
    out_dim: int


def _dims_for(seq_len: int, bg_mode: str) -> tuple[int, int]:
    if bg_mode == "subtract":
        return seq_len, seq_len
    if bg_mode == "subtract_concat":
        return seq_len * 4, seq_len
    if bg_mode == "concat":
        return (seq_len + 1) * 3, seq_len
    return seq_len * 3, seq_len


def _load_raw(path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected a checkpoint dict, got {type(ckpt).__name__}")
    return ckpt


def read_config(path: str | Path) -> TrackNetConfig:
    """Read the TrackNet configuration stored in (or implied by) a checkpoint."""
    ckpt = _load_raw(path)
    state = ckpt.get("model", ckpt)
    in_dim = state["down_block_1.conv_1.conv.weight"].shape[1]
    out_dim = state["predictor.weight"].shape[0]

    param_dict = ckpt.get("param_dict", {})
    seq_len = param_dict.get("seq_len", out_dim)
    bg_mode = param_dict.get("bg_mode")
    if bg_mode is None:
        expected = {mode: _dims_for(seq_len, mode) for mode in ("", "subtract", "subtract_concat", "concat")}
        matches = [mode for mode, dims in expected.items() if dims == (in_dim, out_dim)]
        bg_mode = matches[0] if matches else ""
    return TrackNetConfig(seq_len=seq_len, bg_mode=bg_mode, in_dim=in_dim, out_dim=out_dim)


def load_tracknet(path: str | Path) -> tuple[TrackNet, TrackNetConfig]:
    """Instantiate a TrackNet with weights from `path`. Returns (model.eval(), config)."""
    config = read_config(path)
    ckpt = _load_raw(path)
    model = TrackNet(in_dim=config.in_dim, out_dim=config.out_dim)
    model.load_state_dict(ckpt.get("model", ckpt))
    return model.eval(), config


def load_inpaintnet(path: str | Path) -> InpaintNet:
    """Instantiate an InpaintNet with weights from `path`."""
    ckpt = _load_raw(path)
    model = InpaintNet()
    model.load_state_dict(ckpt.get("model", ckpt))
    return model.eval()
