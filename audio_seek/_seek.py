import os
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

# WAV/AIFF-family subtypes libsndfile can seek into at a fixed byte offset
# per sample (true O(1)). Anything else (FLAC, Vorbis, MPEG, ...) requires
# libsndfile to walk frame boundaries or entropy-decode, so seeking is best
# described as O(n) in the worst case even though the API is the same.
_O1_SUBTYPES = {
    "PCM_S8",
    "PCM_U8",
    "PCM_16",
    "PCM_24",
    "PCM_32",
    "FLOAT",
    "DOUBLE",
    "ALAW",
    "ULAW",
}


class AudioSeek:
    """Read arbitrary time slices of an audio file without loading it fully."""

    @staticmethod
    def get_duration(file_path: Path | str) -> float:
        """
        Gets audio file total duration (seconds).
        Only reads the header, doesn't load audio data (O(1)).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with sf.SoundFile(file_path) as f:
            return f.frames / f.samplerate

    @staticmethod
    def read_segment(
        file_path: Path | str,
        start_sec: float,
        duration_sec: float,
    ) -> np.ndarray:
        from audio_seek.read_audio_segment import read_audio_segment

        return read_audio_segment(file_path, start_sec, duration_sec)

    @staticmethod
    def read_segment_to_file(
        file_path: Path | str,
        start_sec: float,
        duration_sec: float,
        output_path: Path | str,
    ) -> Path:
        """Reads an audio segment and saves it to a PCM WAV (16-bit) file."""
        from audio_seek.read_audio_segment import read_audio_segment

        data = read_audio_segment(file_path, start_sec, duration_sec)

        with sf.SoundFile(file_path) as f:
            sample_rate: int = f.samplerate

        sf.write(output_path, data, sample_rate, format="WAV", subtype="PCM_16")

        return Path(output_path)

    @staticmethod
    def seek_complexity(file_path: Path | str) -> Literal["O(1)", "O(n)"]:
        """
        Reports whether seeking into this file is sample-accurate O(1)
        (uncompressed PCM/float/A-law/mu-law) or O(n) in the worst case
        (FLAC, Vorbis, MPEG, and other frame/entropy-coded formats).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with sf.SoundFile(file_path) as f:
            subtype = f.subtype

        return "O(1)" if subtype in _O1_SUBTYPES else "O(n)"
