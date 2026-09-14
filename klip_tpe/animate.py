"""Assemble the step PNGs into a progress movie (``near2m_makegif`` without
ImageMagick): GIF via imageio (PIL fallback) and MP4 via imageio's ffmpeg backend
when one is available.  Frames are downsampled to ``width`` pixels."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["load_frames", "make_movie", "thin"]


def _pil():
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


def load_frames(paths: Sequence[str], width: int = 900) -> List[np.ndarray]:
    """RGB uint8 frames resized to ``width`` (even dimensions, for the video codecs)."""
    Image = _pil()
    if Image is None:
        raise RuntimeError("Pillow is required (pip install pillow)")
    out: List[np.ndarray] = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w != width:
                nh = int(round(h * width / w))
                im = im.resize((width, nh), Image.LANCZOS)
            w, h = im.size
            if w % 2 or h % 2:
                im = im.crop((0, 0, w - w % 2, h - h % 2))
            out.append(np.asarray(im))
    return out


def thin(paths: Sequence[str], max_frames: int) -> List[str]:
    """At most ``max_frames`` of ``paths``, evenly spaced, always keeping the last one."""
    paths = list(paths)
    if max_frames and len(paths) > max_frames > 0:
        idx = np.unique(np.round(np.linspace(0, len(paths) - 1, int(max_frames))).astype(int))
        paths = [paths[i] for i in idx]
    return paths


def make_movie(paths: Sequence[str], out_gif: str, out_mp4: Optional[str] = None, intro: Optional[str] = None,
               width: int = 900, fps: float = 5.0, intro_frames: int = 8, hold_last: int = 6,
               max_frames: int = 400) -> List[str]:
    """Write ``out_gif`` (and ``out_mp4`` if an ffmpeg backend exists) from ``paths``.
    ``intro`` (a PNG) is repeated ``intro_frames`` times at the start, the last frame
    ``hold_last`` times.  Long runs are thinned to ``max_frames`` evenly spaced frames
    (the last one always kept) so a 10 000-eval annulus stays a few hundred MB of memory
    and a watchable GIF.  Returns the list of files written."""
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return []
    paths = thin(paths, max_frames)
    frames = load_frames(paths, width)
    if intro and os.path.exists(intro):
        fi = load_frames([intro], width)[0]
        h = frames[0].shape[0]
        if fi.shape[0] != h:                   # pad / crop the intro to the frame height
            Image = _pil()
            fi = np.asarray(Image.fromarray(fi).resize((frames[0].shape[1], h), Image.LANCZOS))
        frames = [fi] * int(intro_frames) + frames
    frames = frames + [frames[-1]] * int(hold_last)
    written: List[str] = []
    dur_ms = int(round(1000.0 / fps))
    try:
        import imageio.v3 as iio
        iio.imwrite(out_gif, frames, duration=dur_ms, loop=0)
        written.append(out_gif)
    except Exception:
        Image = _pil()
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(out_gif, save_all=True, append_images=ims[1:], duration=dur_ms, loop=0)
        written.append(out_gif)
    if out_mp4:
        try:
            import imageio.v2 as iio2
            w = iio2.get_writer(out_mp4, fps=fps, codec="libx264", quality=7, macro_block_size=None)
            try:
                for f in frames:
                    w.append_data(f)
            finally:
                w.close()
            written.append(out_mp4)
        except Exception:
            if os.path.exists(out_mp4) and os.path.getsize(out_mp4) == 0:
                os.remove(out_mp4)
    return written
