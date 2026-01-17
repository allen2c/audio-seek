import os
import tempfile
import warnings
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import soundfile as sf

BITS_TYPE: TypeAlias = Literal[2, 3, 4, 5]


class SubtypeInfo(TypedDict):
    """Metadata for a WAV subtype including seekability and bit depth."""

    subtype: str
    seekable: bool
    bits_per_sample: int


# Runtime cache: stores the best seekable subtype for each bit depth
SUBTYPE_CACHE: dict[BITS_TYPE, SubtypeInfo] = {}


class AudioSeek:
    """
    Handles seekable audio WAV file reading, writing and format conversion.
    Automatically selects best available seekable compression format.
    """

    @staticmethod
    def get_duration(file_path: Path | str) -> float:
        """
        Gets audio file total duration (seconds).
        Advantage: Only reads header, doesn't load audio data, extremely fast (O(1)).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # SoundFile object only parses Header when opened
        with sf.SoundFile(file_path) as f:
            # frames = total samples, samplerate = sampling rate
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
    def write(
        file_path: Path | str,
        data: np.ndarray,
        *,
        sample_rate: int = 16000,
        bits_per_sample: BITS_TYPE = 2,
    ) -> Path:
        """
        Writes numpy array to G.726 format.

        Args:
            file_path (str): Output path (recommend .wav)
            data (np.array): Audio data (float32 or int16)
            sample_rate (int): Sampling rate (e.g. 16000, 8000)
            bits_per_sample (int): Compression depth (2, 3, 4, 5).
                                   2 bits @ 16k = 32kbps (recommended)
                                   3 bits @ 16k = 48kbps
        """

        # 1. Check if compression depth is valid
        subtype = AudioSeek.resolve_best_subtype(bits_per_sample)
        if not subtype:
            raise ValueError(
                f"Unsupported bits_per_sample: {bits_per_sample}. "
                + f"Please choose one of: {list(SUBTYPE_CACHE.keys())}"
            )

        # 2. Write file with specified seekable format
        # format='WAV' is container, subtype determines encoding
        sf.write(file_path, data, sample_rate, format="WAV", subtype=subtype)

        return Path(file_path)

    @classmethod
    def convert(
        cls,
        data: np.ndarray,
        output_path: Path | str,
        src_sr: int,
        target_sr: int = 16000,
        bits: "BITS_TYPE" = 2,
        to_mono: bool = True,
    ) -> Path:
        """
        Converts numpy array to G.726 WAV format.

        Args:
            data (np.array): Audio data
            src_sr (int): Source sample rate (required for resampling)
            output_path (Path | str): Output file path
            target_sr (int): Target sample rate (default 16000)
            bits (BITS_TYPE): Compression depth (2, 3, 4, 5)
            to_mono (bool): Whether to force conversion to mono
        """
        import librosa

        from audio_seek.ensure_mono import ensure_mono

        # 1. Handle channels (Mono Mixing)
        if to_mono:
            data = ensure_mono(data, style="librosa")

        # 2. Handle resampling (if source sr != target sr)
        if src_sr != target_sr:
            # Use librosa's resample function
            # Note: This step is CPU intensive
            data = librosa.resample(data, orig_sr=src_sr, target_sr=target_sr)

        # 3. Write file
        subtype = AudioSeek.resolve_best_subtype(bits)
        sf.write(output_path, data, target_sr, format="WAV", subtype=subtype)

        return Path(output_path)

    @classmethod
    def convert_from_file(
        cls,
        input_path: Path | str,
        output_path: Path | str,
        *,
        target_sr: int = 16000,
        bits: "BITS_TYPE" = 2,
        to_mono: bool = True,
    ):
        """
        Converts any audio file (MP3, WAV, FLAC...) to G.726 WAV format.

        Args:
            input_path (str): Source file path
            output_path (str): Output file path
            target_sr (int): Target sample rate (default 16000)
            bits (int): Compression depth (2=32kbps, 3=48kbps...)
            to_mono (bool): Whether to force conversion to mono
        """
        import librosa

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # 1. Load with librosa
        try:
            # y is audio data (float32), sr is actual sample rate read
            y, _ = librosa.load(input_path, sr=target_sr, mono=to_mono)
        except Exception as e:
            raise RuntimeError(f"Failed to load or resample: {e}")

        # 2. Write with best seekable format
        subtype = AudioSeek.resolve_best_subtype(bits)
        sf.write(output_path, y, target_sr, format="WAV", subtype=subtype)

        return output_path

    @staticmethod
    def _test_seekability(subtype: str, sample_rate: int = 16000) -> bool:
        """
        Tests if a WAV subtype supports O(1) seek operations.

        Args:
            subtype: WAV subtype to test (e.g., 'IMA_ADPCM', 'G726_32')
            sample_rate: Sample rate for testing (default: 16000)

        Returns:
            True if the subtype supports seek, False otherwise
        """
        # Create minimal test data (0.1 second)
        test_samples = int(sample_rate * 0.1)
        test_data = np.zeros(test_samples, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Try to write the file
            sf.write(tmp_path, test_data, sample_rate, format="WAV", subtype=subtype)

            # Try to seek
            with sf.SoundFile(tmp_path) as f:
                # Seek to middle of the file
                seek_pos = test_samples // 2
                f.seek(seek_pos)
                # Try to read
                f.read(10, dtype="float32")

            return True

        except (sf.LibsndfileError, Exception):
            return False

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def resolve_best_subtype(bits: BITS_TYPE) -> str:
        """
        Finds best seekable subtype for the given bit depth.
        Automatically downgrades to compatible format with warning if needed.
        Priority: Standard G726 -> IMA/MS ADPCM -> Fallback with warning
        """
        # Check cache first
        if bits in SUBTYPE_CACHE:
            return SUBTYPE_CACHE[bits]["subtype"]

        # Get all available WAV subtypes on current system
        available = sf.available_subtypes("WAV")

        # Define priority candidates for each bit depth
        # Priority: Standard G.726 -> Known seekable formats
        candidates_by_bits: dict[BITS_TYPE, list[str]] = {
            2: ["G726_16", "NMS_ADPCM_16"],
            3: ["G726_24", "NMS_ADPCM_24"],
            4: ["G726_32", "G721_32", "IMA_ADPCM", "MS_ADPCM", "NMS_ADPCM_32"],
            5: ["G726_40", "NMS_ADPCM_40"],
        }

        candidates = candidates_by_bits.get(bits, [])

        # Find first available AND seekable format
        selected_info: SubtypeInfo | None = None

        for candidate in candidates:
            if candidate not in available:
                continue

            # Test if this format supports seek
            is_seekable = AudioSeek._test_seekability(candidate)

            if is_seekable:
                selected_info = SubtypeInfo(
                    subtype=candidate,
                    seekable=True,
                    bits_per_sample=bits,
                )
                break

        # Fallback strategy: if no seekable format found for requested bits
        if selected_info is None:
            warnings.warn(
                f"No seekable {bits}-bit format available on this system. "
                f"Falling back to 4-bit IMA_ADPCM for compatibility.",
                UserWarning,
                stacklevel=2,
            )

            # Try standard fallback formats (widely supported and seekable)
            fallback_candidates = ["IMA_ADPCM", "MS_ADPCM"]

            for fallback in fallback_candidates:
                if fallback in available and AudioSeek._test_seekability(fallback):
                    selected_info = SubtypeInfo(
                        subtype=fallback,
                        seekable=True,
                        bits_per_sample=4,  # These are typically 4-bit
                    )
                    break

        # Last resort: if even fallbacks don't work, raise error
        if selected_info is None:
            raise RuntimeError(
                "Critical error: No seekable ADPCM format available on this system. "
                "Please check your libsndfile installation."
            )

        # Cache the result
        SUBTYPE_CACHE[bits] = selected_info
        return selected_info["subtype"]
