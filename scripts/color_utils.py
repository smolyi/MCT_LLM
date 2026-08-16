"""
Explicit pixel-based color naming for garment regions -- deterministic,
computed from actual crop pixels, not left to a captioning model's guess.

Per explicit user direction: color hallucination/misreading is a known VLM
weak point across model families (not just BLIP), so this is factored out
of the caption model entirely. The chosen captioning model (Decision 1) is
prompted for garment TYPE only ("shirt", "jacket") and explicitly told not
to name colors; extract_entity_attributes.py splices this module's output
in programmatically to build the final appearance_caption -- the same
deterministic-report pattern used everywhere else in this project (facts
rendered by code, never retyped/guessed by the LLM), applied one level
down into caption construction itself instead of only at the
query-answering layer.

Usage:
    from color_utils import garment_region, dominant_color_name
    crop = frame[y1:y2, x1:x2]  # BGR, as read by cv2
    upper_name = dominant_color_name(garment_region(crop, "upper"))
    lower_name = dominant_color_name(garment_region(crop, "lower"))
"""
import cv2
import numpy as np

# Fixed named palette in RGB (0-255) -- basic colors plus a couple of
# shades, not exact hex precision (a query like "red shirt" doesn't need
# to distinguish crimson from scarlet). Matched by nearest CIE Lab
# distance, not raw RGB distance, since Lab distance tracks how colors
# actually look different -- raw RGB distance doesn't (e.g. it
# underweights the difference between navy and black).
NAMED_COLORS_RGB = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "dark gray": (70, 70, 70),
    "red": (200, 30, 30),
    "dark red": (120, 20, 20),
    "orange": (230, 120, 30),
    "yellow": (220, 200, 40),
    "green": (40, 150, 60),
    "dark green": (20, 90, 40),
    "blue": (40, 90, 200),
    "dark blue": (20, 40, 110),
    "light blue": (130, 180, 230),
    "purple": (120, 50, 150),
    "pink": (230, 140, 180),
    "brown": (110, 70, 40),
    "beige": (210, 190, 150),
}


def _rgb_to_lab(rgb: tuple) -> np.ndarray:
    arr = np.uint8([[list(rgb)]])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(float)


_NAMED_LAB = {name: _rgb_to_lab(rgb) for name, rgb in NAMED_COLORS_RGB.items()}


def garment_region(crop_bgr: np.ndarray, part: str) -> np.ndarray:
    """Slices a person crop (BGR, as read by cv2) into a garment region by fixed
    bbox-height fraction, with a side margin to skip background bleeding in at the
    crop's edges. 'upper' ~ torso (skips head/neck above, skips waist blending
    below); 'lower' ~ legs (skips waist above, skips shoes below). These fractions
    are a reasonable starting guess for a standing/walking person, not empirically
    tuned yet -- worth spot-checking against real crops during Phase 1."""
    h, w = crop_bgr.shape[:2]
    if part == "upper":
        y0, y1 = int(h * 0.25), int(h * 0.55)
    elif part == "lower":
        y0, y1 = int(h * 0.60), int(h * 0.90)
    else:
        raise ValueError(f"unknown garment part: {part!r}, expected 'upper' or 'lower'")
    x0, x1 = int(w * 0.15), int(w * 0.85)
    region = crop_bgr[y0:y1, x0:x1]
    return region if region.size else crop_bgr


# Words a captioning VLM might use to describe color, despite being explicitly instructed not to
# (confirmed empirically in the Phase 0 bake-off -- Qwen2.5-VL-3B and BLIP-2 both ignored a "do not
# mention color" instruction in their constrained-prompt output). Stripped from VLM garment-type text
# before splicing in the RGB-sampled color name, so the two can never contradict each other -- a
# deterministic fix rather than relying on the prompt alone, per this project's dominant lesson.
COLOR_STOPWORDS = {
    "black", "white", "gray", "grey", "dark", "light", "red", "orange", "yellow", "green",
    "blue", "purple", "pink", "brown", "beige", "navy", "maroon", "tan", "khaki", "silver",
    "gold", "golden", "olive", "teal", "magenta", "cream", "charcoal", "burgundy",
}


def _is_color_word(token: str) -> bool:
    """A token counts as color-naming if it IS a color word, or is a hyphenated compound containing
    one (e.g. "dark-colored", "red-and-white") -- found necessary after a real miscalibration: the
    word-level check alone let "dark-colored jacket" straight through since "dark-colored" as a whole
    string never matches the COLOR_STOPWORDS set."""
    stripped = token.strip(".,;:").lower()
    if stripped in COLOR_STOPWORDS:
        return True
    return "-" in stripped and any(part in COLOR_STOPWORDS for part in stripped.split("-"))


def strip_color_words(text: str) -> str:
    """Removes color-naming tokens (including hyphenated compounds) from free-form VLM text, collapsing
    extra whitespace left behind. Deterministic complement to the VLM's own (unreliable) "don't mention
    color" instruction -- see COLOR_STOPWORDS."""
    words = text.split()
    kept = [w for w in words if not _is_color_word(w)]
    return " ".join(kept).replace(" ,", ",").replace(" .", ".")


MIN_GARMENT_REGION_PIXELS = 150  # real bug found via direct visual inspection (user-reported): a
# 38x13px crop (494px total area -- enough to PASS crop_quality's coarser area>=400 gate) sliced
# down to a ~72px garment region gave "black" for a crop that was genuinely cream/beige; the SAME
# person's larger crops (580px+ garment region) correctly gave "beige". crop_quality's gate operates
# on the WHOLE crop, not the garment-region sub-slice specifically -- a crop can pass that gate and
# still leave garment_region() with too few pixels for k-means to find a real cluster rather than
# noise. Calibrated against a real 18-crop sample's garment-region sizes (p5=450, p10=519, min
# legitimate-looking case ~165) -- 150 sits just below that real population, comfortably above the
# confirmed-bad ~72px case.


def dominant_color_name(region_bgr: np.ndarray, k: int = 3) -> str:
    """K-means over a garment region's pixels in Lab space (so clustering itself
    tracks perceptual similarity), taking the largest cluster as the dominant color
    -- a crude but effective way to reject small background/skin-tone slivers that
    leak in at the region's edges -- then names it via nearest Lab distance to the
    fixed palette above. Returns "unknown" for a region too small to trust (see
    MIN_GARMENT_REGION_PIXELS) rather than guessing from too little real signal."""
    if region_bgr.size == 0 or region_bgr.shape[0] * region_bgr.shape[1] < MIN_GARMENT_REGION_PIXELS:
        return "unknown"
    lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) < k:
        return "unknown"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant_lab = centers[np.argmax(counts)]
    best_name, best_dist = "unknown", float("inf")
    for name, name_lab in _NAMED_LAB.items():
        dist = float(np.linalg.norm(dominant_lab - name_lab))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name
