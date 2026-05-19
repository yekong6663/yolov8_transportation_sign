#!/usr/bin/env python3
"""
Export YOLOv8 ONNX model to K230 kmodel format.

Requires nncase: pip install nncase nncase-kpu
"""

import argparse
import os
import sys
from pathlib import Path


def check_nncase():
    try:
        import nncase
        return True
    except ImportError:
        print("Error: nncase not installed.")
        print("  Install: pip install nncase nncase-kpu")
        print("  Or download from: https://github.com/kendryte/nncase/releases")
        sys.exit(1)


def generate_calib_data(img_dir: str, input_shape, count: int = 20):
    """Generate calibration data for quantization from images."""
    import numpy as np
    from PIL import Image

    img_dir = Path(img_dir)
    img_files = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
    if not img_files:
        print(f"Warning: No images found in {img_dir}, skipping calibration data.")
        return None

    img_files = img_files[:count]
    _, _, h, w = input_shape
    calib_data = []
    for f in img_files:
        img = Image.open(f).convert("RGB").resize((w, h))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, axis=0)  # add batch dim
        calib_data.append(arr)

    return np.concatenate(calib_data, axis=0)


def compile_onnx_to_kmodel(
    onnx_path: str,
    output_path: str,
    dataset_dir: str = None,
    input_shape=(1, 3, 640, 640),
    quantize: bool = True,
    target: str = "k230",
):
    import nncase

    print(f"[1/5] Reading ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        onnx_content = f.read()

    print(f"[2/5] Setting compile options (target={target}, quantize={quantize})")
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_shape = list(input_shape)
    compile_options.input_layout = "NCHW"
    compile_options.output_layout = "NCHW"
    compile_options.input_type = "float32"
    compile_options.input_range = [0.0, 1.0]
    compile_options.mean = [0.0, 0.0, 0.0]
    compile_options.std = [1.0, 1.0, 1.0]
    compile_options.preprocess = True

    compiler = nncase.Compiler(compile_options)

    print(f"[3/5] Importing ONNX...")
    compiler.import_onnx(onnx_content, nncase.ImportOptions())

    if quantize:
        print(f"[4/5] Setting up quantization...")
        ptq_options = nncase.PTQTensorOptions()
        ptq_options.calibrate_method = "NoClip"
        ptq_options.quant_type = "int8"
        ptq_options.w_quant_type = "int8"
        ptq_options.samples_count = 20

        if dataset_dir:
            print(f"  Generating calibration data from: {dataset_dir}")
            calib_data = generate_calib_data(dataset_dir, input_shape)
            if calib_data is not None:
                calib_list = [calib_data[i : i + 1] for i in range(calib_data.shape[0])]
                ptq_options.set_tensor_data([calib_list])
                print(f"  Calibration samples: {calib_data.shape[0]}")
        else:
            print("  Warning: No calibration dataset provided, using random data.")
            import numpy as np
            rand_data = np.random.randn(20, *input_shape[1:]).astype(np.float32)
            rand_list = [rand_data[i : i + 1] for i in range(rand_data.shape[0])]
            ptq_options.set_tensor_data([rand_list])

        compiler.use_ptq(ptq_options)

    step = "5/5" if quantize else "4/4"
    print(f"[{step}] Compiling and generating kmodel...")
    compiler.compile()
    kmodel = compiler.gencode_tobytes()

    print(f"[{step}] Saving kmodel to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kmodel)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nDone. kmodel saved: {output_path} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 ONNX to K230 kmodel"
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output kmodel (default: same dir as onnx, .kmodel extension)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset directory for calibration images (recommended for int8 quant)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Model input size (default: 640 640)",
    )
    parser.add_argument(
        "--target",
        default="k230",
        choices=["k230", "k510", "cpu"],
        help="Target device (default: k230)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable int8 quantization (output float32 kmodel)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["4c", "10c"],
        help="Use preset paths for 4c or 10c model",
    )
    args = parser.parse_args()

    check_nncase()

    # Resolve paths with presets
    project_root = Path(__file__).resolve().parent.parent

    if args.preset:
        if args.preset == "4c":
            args.onnx = str(
                project_root
                / "runs/detect/traffic_signs_4c/weights/best.onnx"
            )
            args.output = args.output or str(
                project_root
                / "runs/detect/traffic_signs_4c/weights/best.kmodel"
            )
            args.dataset = args.dataset or str(project_root / "dataset_4c/images/train")
        elif args.preset == "10c":
            args.onnx = str(
                project_root / "runs/detect/traffic_signs/weights/best.onnx"
            )
            args.output = args.output or str(
                project_root / "runs/detect/traffic_signs/weights/best.kmodel"
            )
            args.dataset = args.dataset or str(project_root / "dataset/images/train")

    if not args.onnx:
        print("Error: --onnx is required when --preset is not used.")
        sys.exit(1)

    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    if args.output is None:
        args.output = str(Path(args.onnx).with_suffix(".kmodel"))

    input_shape = (1, 3, args.input_size[0], args.input_size[1])

    compile_onnx_to_kmodel(
        onnx_path=args.onnx,
        output_path=args.output,
        dataset_dir=args.dataset,
        input_shape=input_shape,
        quantize=not args.no_quantize,
        target=args.target,
    )


if __name__ == "__main__":
    main()
