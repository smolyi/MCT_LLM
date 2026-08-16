"""
Resolves a camera directory to whatever source its actual video/frame data
is in -- most sources are a real video.mp4, but EPFL-RLC ships JPEG frame
sequences instead (no video file at all). Shared by track_single_camera.py,
extract_track_embeddings.py, and extract_entity_attributes.py so each
doesn't need its own bespoke frames-vs-video detection logic.

Two different resolvers, not one, because ultralytics' model.track() and
cv2.VideoCapture disagree about how they want an image sequence -- found
empirically, not assumed:
- cv2.VideoCapture reads an image sequence via a printf-style glob pattern
  (e.g. "%06d.jpeg") through its image2 demuxer -- but only if the sequence
  is 0-indexed with no gaps; EPFL-RLC's real filenames (e.g.
  RLCAFTCONF-C0_100000.jpeg) start at 100000, which the demuxer doesn't
  autodetect on its own.
- ultralytics' source loader does NOT accept that pattern -- passing it
  raises FileNotFoundError (it checks the literal path, doesn't interpret
  "%06d" as a format string). It wants a plain directory instead, which it
  globs for known image extensions itself -- confirmed working directly.

Both resolvers share the same zero-indexed symlink directory (built once,
idempotently) -- they just return a different string pointing into it.
"""
from pathlib import Path


def resolve_video_source(camera_dir: Path) -> str:
    """For cv2.VideoCapture (extract_track_embeddings.py, extract_entity_attributes.py): the real
    video.mp4 if one exists, otherwise a 0-indexed frame-sequence GLOB PATTERN."""
    camera_dir = Path(camera_dir)
    video_path = camera_dir / "video.mp4"
    if video_path.exists():
        return str(video_path)
    frames_dir = _ensure_zero_indexed_frames(camera_dir)
    return str(frames_dir / "%06d.jpeg")


def resolve_track_source(camera_dir: Path) -> str:
    """For ultralytics' model.track() (track_single_camera.py): the real video.mp4 if one exists,
    otherwise the 0-indexed frame-sequence DIRECTORY itself (not a pattern -- ultralytics globs a
    directory's images on its own and rejects a printf-style pattern outright)."""
    camera_dir = Path(camera_dir)
    video_path = camera_dir / "video.mp4"
    if video_path.exists():
        return str(video_path)
    return str(_ensure_zero_indexed_frames(camera_dir))


def _ensure_zero_indexed_frames(camera_dir: Path) -> Path:
    """Builds camera_dir/frames/000000.jpeg, 000001.jpeg, ... as symlinks into
    camera_dir/frames_raw/'s real files, sorted by filename -- idempotent (skipped if
    already built with the same file count)."""
    raw_frames_dir = camera_dir / "frames_raw"
    if not raw_frames_dir.is_dir():
        raise FileNotFoundError(
            f"{camera_dir} has neither video.mp4 nor a frames_raw/ directory -- "
            f"nothing this project knows how to read frames from."
        )
    frames_dir = camera_dir / "frames"
    raw_files = sorted(raw_frames_dir.iterdir())
    if frames_dir.is_dir() and len(list(frames_dir.iterdir())) == len(raw_files):
        return frames_dir

    frames_dir.mkdir(exist_ok=True)
    for i, f in enumerate(raw_files):
        link = frames_dir / f"{i:06d}.jpeg"
        if not link.exists():
            link.symlink_to(f.resolve())
    return frames_dir
