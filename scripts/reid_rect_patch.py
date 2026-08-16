"""
Monkeypatch for ultralytics' BoT-SORT ReID encoder (ultralytics/trackers/utils/reid.py), applied
at runtime -- NOT a modification of the installed package -- so it stays intact across
ultralytics upgrades and is scoped to this project only.

THREE real incompatibilities found by direct testing before trusting this pipeline, not assumed:
1. ultralytics' `ReID.__init__` only reads the ONNX model's height dim (`shape[2]`) and its
   `_crops_to_tensor` always resizes crops to a SQUARE (imgsz, imgsz) target. This project's OSNet-
   Market1501 checkpoint (scripts/export_osnet_onnx.py) is trained on the standard person-ReID
   RECTANGULAR aspect (256x128, height:width = 2:1, torchreid's own default) -- squishing a crop to
   square before feeding it in would silently distort the input and degrade the very appearance-
   matching quality this whole effort exists to improve, not crash (which would have been easier to
   catch).
2. `_crops_to_tensor` only scales pixels by /255 -- no ImageNet mean/std normalization. This
   project's OSNet export was verified (cosine similarity 0.99998 against the original PyTorch
   model on a real crop) using torchreid's own preprocessing convention, which DOES apply
   ImageNet mean/std normalization -- feeding unnormalized input would silently miscalibrate
   every embedding this tracker produces.
3. (Found AFTER fixing 1 and 2, via direct GPU-utilization monitoring, not assumed fixed just
   because it ran without error): ultralytics' ONNX path goes through `onnxruntime`'s
   CUDAExecutionProvider, which on this machine silently fails to actually engage the GPU (a
   warning -- "No registered plugin EP device found for CUDAExecutionProvider" -- with real GPU
   utilization staying at ~3-4% throughout a run, an onnxruntime-gpu/CUDA-toolkit version mismatch,
   not a code bug). At that rate a single POM camera would take over an hour, making a full 3-scene
   rollout impractical. Routed around entirely by bypassing ultralytics' AutoBackend+ONNX Runtime
   path -- `TorchReIDEncoder` below calls this project's already-confirmed-fast torchreid/PyTorch/
   CUDA `FeatureExtractor` directly (the exact same code path `extract_track_embeddings.py` already
   uses successfully throughout this project), never touching ONNX Runtime at all.

Usage: import this module BEFORE constructing any BoT-SORT tracker with with_reid=True, e.g. at
the top of track_single_camera.py. Patches apply globally to the ultralytics.trackers.utils.reid
and ultralytics.trackers.bot_sort modules for the lifetime of the process. Point the tracker
YAML's `model:` field at the .pth torchreid checkpoint directly (not the .onnx export) to route
through TorchReIDEncoder instead of the ONNX path.
"""
import numpy as np
import torch

from ultralytics.trackers.utils import reid as _reid_module

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _patched_init(self, model: str, imgsz: int = 224, device=None, fp16: bool = False):
    self.imgsz = imgsz
    self.imgsz_h, self.imgsz_w = imgsz, imgsz  # overwritten below once the real model shape is known
    self.batch_size = None
    self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.is_pt = str(model).endswith(".pt")

    if self.is_pt:
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.model(embed=[len(self.model.model.model) - 2], device=self.device, verbose=False, save=False)
        self.fp16 = False
    else:
        from pathlib import Path
        from ultralytics.nn.autobackend import AutoBackend

        if Path(str(model)).name in _reid_module.REID_ASSETS:
            from ultralytics.utils.downloads import attempt_download_asset
            model = attempt_download_asset(str(model))
        self.model = AutoBackend(str(model), device=self.device, fp16=fp16, verbose=False)
        self.fp16 = self.model.fp16

        session = getattr(self.model, "session", None)
        shape = session.get_inputs()[0].shape if session is not None else ()
        if len(shape) == 4:
            if isinstance(shape[0], int) and shape[0] > 0:
                self.batch_size = shape[0]
            if isinstance(shape[2], int) and shape[2] > 0:
                self.imgsz_h = shape[2]
                self.imgsz = shape[2]  # kept in sync for any code that still reads .imgsz directly
            if isinstance(shape[3], int) and shape[3] > 0:
                self.imgsz_w = shape[3]


def _patched_crops_to_tensor(self, crops: list) -> torch.Tensor:
    """Same as the original, but resizes to (imgsz_h, imgsz_w) -- NOT forced square -- and applies
    ImageNet mean/std normalization, matching torchreid's own preprocessing convention that this
    project's OSNet export was verified against."""
    imgsz_h = getattr(self, "imgsz_h", self.imgsz)
    imgsz_w = getattr(self, "imgsz_w", self.imgsz)
    batch = torch.empty(len(crops), 3, imgsz_h, imgsz_w, dtype=torch.float32)
    for i, c in enumerate(crops):
        t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        batch[i] = torch.nn.functional.interpolate(
            t, size=(imgsz_h, imgsz_w), mode="bilinear", align_corners=False
        )[0]
    batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD
    batch = batch.to(self.device)
    return batch.half() if self.fp16 else batch


class TorchReIDEncoder:
    """Drop-in replacement for ultralytics' ReID callable, `(img, dets) -> list[np.ndarray | None]`,
    that uses this project's own torchreid FeatureExtractor (PyTorch/CUDA) directly -- bypassing
    ONNX Runtime entirely, since its CUDAExecutionProvider was confirmed (via nvidia-smi, not
    assumed) to silently fall back to near-CPU-speed inference on this machine. Same crop-extraction
    convention as ultralytics' own ReID class (xywh->xyxy->save_one_box), just fed into a different
    (already-proven-fast-and-correct-in-this-project) backend."""

    def __init__(self, checkpoint_path: str, device: str | torch.device | None = None):
        from torchreid.reid.utils import FeatureExtractor
        device = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = FeatureExtractor(model_name="osnet_x1_0", model_path=checkpoint_path, device=device)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list:
        from ultralytics.trackers.utils.reid import ReID
        crops = ReID._crop_detections(img, dets)
        valid_idxs = [i for i, c in enumerate(crops) if c.size]
        if not valid_idxs:
            return [None] * len(crops)
        # torchreid's FeatureExtractor treats numpy arrays as RGB with no conversion (same
        # BGR->RGB flip already documented/required in extract_track_embeddings.py).
        rgb_crops = [crops[i][:, :, ::-1] for i in valid_idxs]
        feats = self.fe(rgb_crops).cpu().numpy()
        out = [None] * len(crops)
        for out_i, feat in zip(valid_idxs, feats):
            out[out_i] = feat
        return out


_TORCHREID_MODEL_SUFFIXES = (".pth", ".pth.tar")


def _patched_build_encoder(with_reid: bool, model, device=None):
    if not with_reid:
        return None
    if isinstance(model, str) and model.endswith(_TORCHREID_MODEL_SUFFIXES):
        return TorchReIDEncoder(model, device=device)
    return _original_build_encoder(with_reid, model, device)


def apply():
    global _original_build_encoder
    _reid_module.ReID.__init__ = _patched_init
    _reid_module.ReID._crops_to_tensor = _patched_crops_to_tensor
    _original_build_encoder = _reid_module.build_encoder
    _reid_module.build_encoder = _patched_build_encoder
    # bot_sort.py did `from .utils.reid import build_encoder` at import time, binding its own
    # module-level name -- patching reid_module.build_encoder alone doesn't reach that binding, so
    # it's patched directly too.
    from ultralytics.trackers import bot_sort as _bot_sort_module
    _bot_sort_module.build_encoder = _patched_build_encoder


apply()
