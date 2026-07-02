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


def _linear_resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Lightweight numpy-only linear-interpolation resample. Lower quality
    than a dedicated resampler (e.g. polyphase/sinc), but keeps this
    library dependency-free; resample upstream with a dedicated library
    if you need higher fidelity.
    """
    if orig_sr == target_sr or len(data) == 0:
        return data

    target_length = int(round(len(data) * target_sr / orig_sr))
    orig_indices = np.arange(len(data))
    target_indices = np.linspace(0, len(data) - 1, num=target_length)

    if data.ndim == 1:
        return np.interp(target_indices, orig_indices, data).astype(data.dtype)

    channels = [
        np.interp(target_indices, orig_indices, data[:, ch])
        for ch in range(data.shape[1])
    ]
    return np.stack(channels, axis=1).astype(data.dtype)


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
        *,
        format: str | None = None,
        subtype: str | None = None,
        sample_rate: int | None = None,
    ) -> Path:
        """
        Reads an audio segment and saves it to a file.

        Args:
            format: Output container format (e.g. "WAV", "FLAC", "OGG").
                Defaults to inferring from output_path's extension.
            subtype: Output encoding subtype (e.g. "PCM_16", "PCM_24",
                "FLOAT", "VORBIS"). Defaults to the format's standard
                subtype.
            sample_rate: Output sample rate. Defaults to the source
                file's sample rate. If different, the segment is
                resampled via lightweight linear interpolation (see
                _linear_resample).
        """
        from audio_seek.read_audio_segment import read_audio_segment

        data = read_audio_segment(file_path, start_sec, duration_sec)

        with sf.SoundFile(file_path) as f:
            source_sample_rate: int = f.samplerate

        output_sample_rate = sample_rate or source_sample_rate
        if output_sample_rate != source_sample_rate:
            data = _linear_resample(data, source_sample_rate, output_sample_rate)

        resolved_format = format or Path(output_path).suffix.lstrip(".").upper() or None

        if subtype is not None:
            if resolved_format is None:
                raise ValueError(
                    "Cannot validate subtype without a resolvable format; "
                    "pass format= explicitly or use an output_path with a "
                    "recognized extension."
                )
            if not sf.check_format(resolved_format, subtype):
                raise ValueError(
                    f"Unsupported subtype {subtype!r} for format "
                    f"{resolved_format!r}. Available subtypes: "
                    f"{list(sf.available_subtypes(resolved_format))}"
                )
        elif (
            resolved_format is not None
            and resolved_format not in sf.available_formats()
        ):
            raise ValueError(
                f"Unsupported format {resolved_format!r}. "
                f"Available formats: {list(sf.available_formats())}"
            )

        sf.write(output_path, data, output_sample_rate, format=format, subtype=subtype)

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
