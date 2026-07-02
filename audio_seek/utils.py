"""Utility helpers for audio_seek that aren't part of the core public API."""

import random
from pathlib import Path

import numpy as np


def random_peek_segment(
    file_path: Path | str,
    duration_sec: float,
    *,
    seed: int | None = None,
) -> tuple[float, np.ndarray]:
    """
    Randomly samples a segment of at most duration_sec from file_path.

    The start point is drawn uniformly from [0, max(0, total_duration -
    duration_sec)], so the segment never runs past end-of-file and the
    distribution isn't skewed toward the tail. If the file is shorter
    than duration_sec, the entire file is returned (start_sec == 0.0).

    Note: for compressed/entropy-coded formats (mp3, FLAC, OGG/Vorbis),
    seeking to the chosen start point is O(n) (see AudioSeek.seek_complexity),
    so this can be noticeably slower on large compressed files than on PCM.

    Args:
        file_path: Audio file to peek into.
        duration_sec: Desired segment length in seconds. Must be > 0.
        seed: Optional seed for reproducible sampling (useful in tests).

    Returns:
        (start_sec, data): the actual start offset chosen, and the
        decoded float32 segment. `data` may be shorter than
        `duration_sec` implies if the source file itself is shorter.

    Raises:
        ValueError: if duration_sec <= 0.
        FileNotFoundError: if file_path doesn't exist.
    """
    from audio_seek._seek import AudioSeek

    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")

    total_duration = AudioSeek.get_duration(file_path)

    max_start = max(0.0, total_duration - duration_sec)
    start_sec = random.Random(seed).uniform(0.0, max_start)

    data = AudioSeek.read_segment(file_path, start_sec, duration_sec)

    return start_sec, data
