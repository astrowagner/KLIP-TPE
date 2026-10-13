"""Files that survive a machine crash, not only an interrupted process.

Writing a temporary file and renaming it over the old one makes a write atomic.  A reader
sees the old file or the new one, never half of one.  It does not make the write durable.
The data wait in memory until the system writes them back, about 30 s later, and the
rename can reach the disk first.  When patrick crashed hard during
``run_20261011_145111``, every file written in its last ~30 s came back with 0 bytes,
``checkpoint.json`` among them, and the run had nothing to resume from.

The cure is the standard one, and it costs little at the rate these files are written.
:func:`write` fsyncs the data before the rename and the directory after it, so the rename
is on disk too when it returns.  An append-only log such as ``results.jsonl`` is fsynced
(:func:`fsync_file`) before any file that counts its lines is written.  On macOS ``fsync``
stops at the drive's own cache, so the data file of :func:`write` also gets
``F_FULLFSYNC`` there.

Where a file system cannot do this, a write stays as atomic as it was and carries on.
Windows cannot open a directory, and some network and FUSE mounts refuse fsync.
"""
from __future__ import annotations

import errno
import os
import tempfile
from typing import Optional, Union

try:
    import fcntl
    _FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", None)
except ImportError:                      # pragma: no cover  (Windows)
    fcntl = None
    _FULLFSYNC = None

__all__ = ["fsync_fd", "fsync_file", "fsync_dir", "write", "append_line", "repair_tail"]

#: errors that mean "this file system does not do fsync", not "the write failed"
_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}


def fsync_fd(fd: int, full: bool = False) -> bool:
    """fsync an open descriptor.  ``full`` adds ``F_FULLFSYNC`` on macOS, so the drive
    empties its cache too.  Returns False when the file system does not support fsync.  A
    real failure (``EIO``, ``ENOSPC``) raises."""
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED:
            return False
        raise
    if full and _FULLFSYNC is not None:
        try:
            fcntl.fcntl(fd, _FULLFSYNC)
        except OSError:
            pass
    return True


def fsync_file(path: str) -> bool:
    """Make everything written to ``path`` so far durable, for a log that another call
    appended to and closed.  Never raises.  Returns False when the file is gone or cannot
    be synced."""
    # read-only first (enough everywhere but Windows, which wants a writable descriptor)
    for flags in (os.O_RDONLY, os.O_WRONLY | os.O_APPEND):
        try:
            fd = os.open(path, flags)
        except OSError:
            return False
        try:
            return fsync_fd(fd)
        except OSError:
            continue
        finally:
            os.close(fd)
    return False


def fsync_dir(path: str) -> bool:
    """Make the entries of directory ``path`` durable (a file renamed into it or created in
    it).  Never raises.  Returns False where a directory cannot be synced, as on Windows."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return False
    try:
        os.fsync(fd)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def write(path: str, data: Union[str, bytes], keep: Optional[str] = None) -> None:
    """Replace ``path`` with ``data`` so that, whenever the machine stops, the file holds
    either the old contents or all of the new.

    ``keep`` names where the version being replaced goes first, so the save before the
    newest stays on disk as well.  The move comes after the new data are durable, so a
    crash always leaves one complete version.  In the instant between the two renames
    ``path`` is missing and ``keep`` holds the old one.
    """
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        fsync_dir(os.path.dirname(d))    # the new directory's own entry
    binary = isinstance(data, (bytes, bytearray, memoryview))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=os.path.splitext(path)[1] or ".tmp")
    try:
        with os.fdopen(fd, "wb" if binary else "w") as f:
            f.write(data)
            f.flush()
            fsync_fd(f.fileno(), full=True)
        if keep is not None:
            try:
                os.replace(path, keep)
            except FileNotFoundError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    fsync_dir(d)


def append_line(path: str, line: str) -> None:
    """Append ``line`` to a log on a line of its own.  A crash can leave a log ending in
    half a line, or in zeros, and a line written straight after it would fuse with it and
    be lost to every reader.  Starting a new line first confines the damage to the half
    line."""
    with open(path, "ab+") as f:
        sep = b""
        if f.seek(0, os.SEEK_END) > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                sep = b"\n"
        f.write(sep + line.rstrip("\n").encode() + b"\n")


def repair_tail(path: str) -> str:
    """Cut a torn last line off an append-only log, everything after its last newline.
    Returns a phrase for the log, or "" when the file ended cleanly.

    Only for logs whose lines a checkpoint counts and which were fsynced before that
    checkpoint was written (:func:`fsync_file`).  Their torn tail then always lies beyond
    the last line any checkpoint counted, in an evaluation the resumed search runs and
    logs again.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size == 0:
        return ""
    with open(path, "rb+") as f:
        end, keep = size, 0
        while end > 0:
            start = max(0, end - 65536)
            f.seek(start)
            nl = f.read(end - start).rfind(b"\n")
            if nl >= 0:
                keep = start + nl + 1
                break
            end = start
        if keep == size:
            return ""
        f.truncate(keep)
        f.flush()
        fsync_fd(f.fileno())
    return f"cut {size - keep} byte(s) after its last complete line, left by a write the machine did not finish"
