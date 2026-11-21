"""Assemble the step panels into a progress movie, a GIF and an MP4.

It ports the IDL predecessor's ``near2m_makegif`` without ImageMagick.  The GIF goes through imageio,
with Pillow as the fallback, and the MP4 through imageio's ffmpeg backend when one is installed.  The
frames are resized to ``width`` pixels.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["load_frames", "make_movie", "thin"]


def _pil():
    """Return Pillow's ``Image`` module.

    Returns
    -------
    image : module or None
        ``PIL.Image``, None when Pillow is missing.
    """
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


def load_frames(paths: Sequence[str], width: int = 900) -> List[np.ndarray]:
    """Read images as RGB frames of one width, with even sides for the video codecs.

    Parameters
    ----------
    paths : sequence of str
        The images.
    width : int, default 900
        The frames' width, in pixels.

    Returns
    -------
    frames : list of ndarray, shape (h, w, 3)
        The frames, uint8, each cropped by a pixel where a side is odd.

    Raises
    ------
    RuntimeError
        If Pillow is not installed.
    """
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
    """Return at most ``max_frames`` paths, evenly spaced, the last always kept.

    Parameters
    ----------
    paths : sequence of str
        The paths.
    max_frames : int
        Number of paths at most, 0 or less for all.

    Returns
    -------
    kept : list of str
        The paths kept, in order.
    """
    paths = list(paths)
    if max_frames and len(paths) > max_frames > 0:
        idx = np.unique(np.round(np.linspace(0, len(paths) - 1, int(max_frames))).astype(int))
        paths = [paths[i] for i in idx]
    return paths


def make_movie(paths: Sequence[str], out_gif: str, out_mp4: Optional[str] = None, intro: Optional[str] = None,
               width: int = 900, fps: float = 5.0, intro_frames: int = 8, hold_last: int = 6,
               max_frames: int = 400) -> List[str]:
    """Write a progress movie from step panels.

    Parameters
    ----------
    paths : sequence of str
        The panels, in order.  Missing ones are skipped.
    out_gif : str
        Path of the GIF.
    out_mp4 : str or None, default None
        Path of the MP4, None for none.  It is written only when imageio has an ffmpeg backend.
    intro : str or None, default None
        An image shown first, None for none.
    width : int, default 900
        The frames' width, in pixels.
    fps : float, default 5.0
        Frames per second.
    intro_frames : int, default 8
        Number of frames the intro is shown for.
    hold_last : int, default 6
        Number of frames the last panel is held for.
    max_frames : int, default 400
        Number of panels at most, evenly spaced with the last kept, so a 10,000-evaluation annulus
        stays a few hundred MB of memory and a watchable GIF.

    Returns
    -------
    written : list of str
        The files written, empty without panels.

    Notes
    -----
    An MP4 that fails part way is removed.  Without imageio-ffmpeg, imageio hands ``.mp4`` to its TIFF
    writer, which used to leave a broken file behind.
    """
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
        # The FFMPEG plugin by name: without imageio-ffmpeg, imageio otherwise hands ".mp4"
        # to its TIFF writer, which writes an 8-byte header and then fails on ``fps``,
        # leaving a broken opt_steps.mp4 behind.
        opened = False
        try:
            import imageio.v2 as iio2
            w = iio2.get_writer(out_mp4, format="FFMPEG", fps=fps, codec="libx264", quality=7,
                                macro_block_size=None)
            opened = True
            try:
                for f in frames:
                    w.append_data(f)
            finally:
                w.close()
            written.append(out_mp4)
        except Exception:
            if os.path.exists(out_mp4) and (opened or os.path.getsize(out_mp4) == 0):
                os.remove(out_mp4)
    return written
