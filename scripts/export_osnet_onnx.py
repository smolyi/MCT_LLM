"""
One-time export: converts the project's already-vetted OSNet-Market1501
torchreid checkpoint (checkpoints/osnet_x1_0_market1501.pth) into ONNX, so
it can be plugged into ultralytics' BoT-SORT tracker as a custom ReID
model (BoT-SORT's `model:` field accepts any AutoBackend-loadable format --
.onnx, .torchscript, etc. -- via ultralytics/trackers/utils/reid.py).

Deliberately reuses this checkpoint rather than ultralytics' own bundled
`yolo26-reid.onnx` asset -- same reasoning already documented in CLAUDE.md
for why OSNet-Market1501 was chosen over that asset in the first place
(its training data provenance isn't publicly documented, so it couldn't be
verified against this project's dataset-ethics convention).

Usage:
  python scripts/export_osnet_onnx.py
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "osnet_x1_0_market1501.pth"
OUT_PATH = PROJECT_ROOT / "checkpoints" / "osnet_x1_0_market1501.onnx"
IMAGE_SIZE = (256, 128)  # torchreid's own default for osnet_x1_0 (height, width)


def main():
    from torchreid.reid.utils import FeatureExtractor

    fe = FeatureExtractor(model_name="osnet_x1_0", model_path=str(CHECKPOINT), device="cpu")
    model = fe.model
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    torch.onnx.export(
        model, dummy, str(OUT_PATH),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {OUT_PATH}")


if __name__ == "__main__":
    main()
