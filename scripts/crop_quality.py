"""
Deterministic crop-quality gate, applied BEFORE any captioning-VLM call.

Motivated by a real finding from the Phase 0 captioning-model bake-off: none
of the three real candidate VLMs (Qwen2.5-VL-3B, LLaVA-OneVision, BLIP-2)
reliably honored an explicit "if unclear, say so" prompt hedge on a
genuinely bad crop -- two of the three confidently invented a specific
color/garment instead. Same lesson as every other hallucination fix in this
project: don't trust the model to self-report uncertainty, gate it in code.

First calibration attempt (real bug, corrected here, not silently overwritten):
sampled 1395 crops from RANDOM (camera, frame) pairs across pred_full.txt and
set thresholds off that distribution's ~p5 (width>=18, area>=1200). That
population is NOT what this pipeline actually samples, though --
extract_entity_attributes.py picks crops spread across each ENTITY's own
trajectory (favoring central, non-margin frames), which is a measurably
different, smaller-skewed distribution (real area p5=364 vs. the random-frame
sample's p5=1135). Applied blindly, the first thresholds flagged 41% of all
934 real POM entities as "unclear" -- caught by comparing the actual full-run
output rate against the calibration's expected ~5%, a big enough gap to be a
real bug, not noise.

Re-diagnosed by pulling real borderline crops (width 10-22px) from the ACTUAL
pipeline and visually inspecting them (same direct-inspection method that
caught the "sandwich" caption bug): a 16x185px crop (area 2960) -- narrow
only because the person was captured mostly upright in a tight bbox -- was a
perfectly legible dark-top/light-pants figure the WIDTH-only rule rejected
outright; a 14x19px crop (area 266) was genuine unreadable mush; the bake-off's
original 8x112px failure case (area 896) sat in between on area alone,
confirming area and width can't fully separate "real narrow crop" from "false-
positive sliver" from geometry alone -- that failure mode is really about
CONTENT (a solid-color blob with no structure), not size, and no purely
geometric rule catches it perfectly. Given that ceiling, the gate is
deliberately kept conservative (favoring false negatives over rejecting
legitimate crops en masse, the same "soft preference over hard exclusion"
lesson already learned once from the frame-margin-crop bug) rather than
tuned to catch every outlier: MIN_CROP_AREA_PX=400 sits just above the
confirmed-bad 266/290px-area examples and below the confirmed-legible 624px
one; the width-only rule is dropped entirely (it produced the false
rejection above); Laplacian variance is kept at its original conservative
value since it targets a different, still-real failure mode (motion blur/
out-of-focus), even though it didn't happen to catch the bake-off's specific
sliver case (whose variance was, misleadingly, above the dataset median).
"""
import cv2
import numpy as np

MIN_CROP_AREA_PX = 400  # see recalibration note above -- just above confirmed-bad ~266-290px-area
# real crops, below a confirmed-legible ~624px-area one, from direct visual inspection.
MIN_LAPLACIAN_VARIANCE = 400  # unchanged -- targets blur specifically, a separate failure mode.

UNCLEAR_CAPTION = "appearance unclear (low-quality crop)"


def is_low_quality_crop(crop_bgr: np.ndarray) -> bool:
    """True if a crop is too small/degenerate or too blurry to trust a captioning
    model's output on -- callers should skip the VLM entirely and use UNCLEAR_CAPTION
    instead of risking a confident guess on unreadable pixels."""
    if crop_bgr.size == 0:
        return True
    h, w = crop_bgr.shape[:2]
    if (w * h) < MIN_CROP_AREA_PX:
        return True
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < MIN_LAPLACIAN_VARIANCE
