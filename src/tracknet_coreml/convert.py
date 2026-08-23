"""Convert TrackNetV3 checkpoints to Core ML mlprogram packages for the Neural Engine.

    tracknet-convert --ckpt TrackNet_best.pt -o TrackNet.mlpackage
    tracknet-convert --ckpt InpaintNet_best.pt --kind inpaintnet -o InpaintNet.mlpackage

Models are exported as fp16 mlprogram with fixed input shapes (the ANE-friendly
configuration) and verified against the fp32 PyTorch reference after conversion.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from tracknet_coreml.checkpoint import load_inpaintnet, load_tracknet

INPUT_H, INPUT_W = 288, 512
INPAINT_SEQ_LEN = 16


def _ct():
    try:
        import coremltools as ct
    except ImportError as e:
        raise SystemExit("coremltools is required: pip install 'tracknet-coreml[convert]'") from e
    return ct


def _annotate(mlmodel, description: str, extra: dict[str, str]) -> None:
    mlmodel.short_description = description
    mlmodel.author = "tracknet-coreml (weights: qaz812345/TrackNetV3)"
    mlmodel.license = "MIT"
    for key, value in extra.items():
        mlmodel.user_defined_metadata[key] = value


def convert_tracknet(ckpt_path: str | Path, out_path: Path):
    ct = _ct()
    model, config = load_tracknet(ckpt_path)
    example = torch.rand(1, config.in_dim, INPUT_H, INPUT_W)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frames", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="heatmaps", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16,
    )
    _annotate(
        mlmodel,
        f"TrackNetV3 ball-tracking heatmap net "
        f"({config.in_dim}ch -> {config.out_dim}ch @ {INPUT_H}x{INPUT_W})",
        {"seq_len": str(config.seq_len), "bg_mode": config.bg_mode,
         "input_layout": "N x C x H x W, RGB frames stacked on C, values in [0, 1]"},
    )
    mlmodel.save(str(out_path))
    return mlmodel, model, (example,)


def convert_inpaintnet(ckpt_path: str | Path, out_path: Path):
    ct = _ct()
    model = load_inpaintnet(ckpt_path)
    coords = torch.rand(1, INPAINT_SEQ_LEN, 2)
    mask = torch.randint(0, 2, (1, INPAINT_SEQ_LEN, 1)).float()
    traced = torch.jit.trace(model, (coords, mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="coords", shape=coords.shape, dtype=np.float32),
            ct.TensorType(name="miss_mask", shape=mask.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="inpainted_coords", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16,
    )
    _annotate(
        mlmodel,
        f"TrackNetV3 InpaintNet trajectory rectifier (seq_len={INPAINT_SEQ_LEN}, "
        "normalized xy coordinates + miss mask)",
        {"seq_len": str(INPAINT_SEQ_LEN)},
    )
    mlmodel.save(str(out_path))
    return mlmodel, model, (coords, mask)


def verify(mlmodel, torch_model: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...]) -> dict:
    """Compare the converted model against the fp32 PyTorch reference on the example input."""
    with torch.no_grad():
        ref = torch_model(*example_inputs).numpy()
    input_names = [inp.name for inp in mlmodel.get_spec().description.input]
    prediction = mlmodel.predict(
        {name: x.numpy().astype(np.float32) for name, x in zip(input_names, example_inputs)}
    )
    out = next(iter(prediction.values()))
    diff = np.abs(ref - out)
    argmax_match = bool(
        (ref.reshape(ref.shape[0], ref.shape[1], -1).argmax(-1)
         == out.reshape(out.shape[0], out.shape[1], -1).argmax(-1)).all()
    ) if ref.ndim == 4 else None
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "heatmap_argmax_identical": argmax_match}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--kind", choices=["tracknet", "inpaintnet"], default="tracknet")
    parser.add_argument("-o", "--out", help="output .mlpackage path (default: <ckpt stem>.mlpackage)")
    parser.add_argument("--no-verify", action="store_true", help="skip the post-conversion parity check")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(args.ckpt).with_suffix(".mlpackage")
    converter = convert_tracknet if args.kind == "tracknet" else convert_inpaintnet
    mlmodel, torch_model, example_inputs = converter(args.ckpt, out_path)
    print(f"saved {out_path}")

    if not args.no_verify:
        report = verify(mlmodel, torch_model, example_inputs)
        print(f"parity vs fp32 PyTorch: max|diff|={report['max_abs_diff']:.5f} "
              f"mean|diff|={report['mean_abs_diff']:.7f}", end="")
        if report["heatmap_argmax_identical"] is not None:
            print(f"  argmax identical: {report['heatmap_argmax_identical']}")
        else:
            print()


if __name__ == "__main__":
    main()
