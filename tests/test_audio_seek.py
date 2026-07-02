"""Tests for audio_seek.AudioSeek (audio_seek/_seek.py)."""

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import SampleAudioData

from audio_seek import AudioSeek


class TestAudioSeek:
    """Test suite for the AudioSeek class."""

    def test_get_duration_accuracy(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Test that get_duration returns an accurate duration."""
        duration = AudioSeek.get_duration(test_wav_file)

        expected = sample_audio_data["duration"]
        assert abs(duration - expected) < 0.05

    def test_get_duration_fast(self, test_wav_file: Path) -> None:
        """Test that get_duration is fast (doesn't load audio data)."""
        start: float = time.time()
        duration = AudioSeek.get_duration(test_wav_file)
        elapsed: float = time.time() - start

        assert duration > 0
        assert elapsed < 0.1  # Should be instant

    def test_get_duration_nonexistent_file(self) -> None:
        """Test that get_duration raises error for non-existent files."""
        with pytest.raises(FileNotFoundError):
            AudioSeek.get_duration("nonexistent_file.wav")

    def test_read_segment_delegates_to_read_audio_segment(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Test that AudioSeek.read_segment matches read_audio_segment."""
        segment = AudioSeek.read_segment(test_wav_file, start_sec=0.2, duration_sec=0.3)

        expected_samples = int(0.3 * sample_audio_data["sample_rate"])
        assert len(segment) == expected_samples
        assert segment.dtype == np.float32

    def test_read_segment_to_file_supports_path_types(
        self, temp_dir: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Test that read_segment_to_file accepts both str and Path objects."""
        source_path = temp_dir / "path_types_source.wav"
        sf.write(
            source_path,
            sample_audio_data["data"],
            sample_audio_data["sample_rate"],
            format="WAV",
            subtype="PCM_16",
        )

        # Test with Path
        path_obj = temp_dir / "path_object.wav"
        result1 = AudioSeek.read_segment_to_file(
            source_path, 0.0, 0.1, output_path=path_obj
        )
        assert result1.exists()

        # Test with str
        str_path: str = str(temp_dir / "str_path.wav")
        result2 = AudioSeek.read_segment_to_file(
            source_path, 0.0, 0.1, output_path=str_path
        )
        assert result2.exists()

    def test_seek_complexity_pcm_wav_is_o1(self, test_wav_file: Path) -> None:
        """Uncompressed PCM WAV should report O(1) seek complexity."""
        assert AudioSeek.seek_complexity(test_wav_file) == "O(1)"

    def test_seek_complexity_nonexistent_file(self) -> None:
        """seek_complexity raises for non-existent files."""
        with pytest.raises(FileNotFoundError):
            AudioSeek.seek_complexity("nonexistent_file.wav")

    def test_get_duration_on_mp3(self, sample_clip_file: Path) -> None:
        """Duration should be about the 15s the sample clip was cut to."""
        duration = AudioSeek.get_duration(sample_clip_file)
        assert 14.0 < duration < 16.0

    def test_seek_complexity_mp3_is_on(self, sample_clip_file: Path) -> None:
        """mp3 is frame/entropy coded, so seeking is not O(1) here."""
        assert AudioSeek.seek_complexity(sample_clip_file) == "O(n)"

    def test_read_segment_from_mp3_middle(self, sample_clip_file: Path) -> None:
        """read_segment can pull a slice from the middle of the mp3 clip."""
        segment = AudioSeek.read_segment(
            sample_clip_file, start_sec=5.0, duration_sec=3.0
        )

        assert segment.dtype == np.float32
        assert len(segment) > 0

    def test_read_segment_to_file_from_mp3(
        self, sample_clip_file: Path, tmp_path: Path
    ) -> None:
        """read_segment_to_file converts a slice of the mp3 into a PCM WAV."""
        output_path = tmp_path / "converted_segment.wav"

        result_path = AudioSeek.read_segment_to_file(
            sample_clip_file,
            start_sec=1.0,
            duration_sec=4.0,
            output_path=output_path,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0
        # The converted output is PCM, so seeking into it is now O(1).
        assert AudioSeek.seek_complexity(result_path) == "O(1)"

    def test_read_segment_to_file_infers_format_from_extension(
        self, test_wav_file: Path, tmp_path: Path
    ) -> None:
        """Without format=, the container is inferred from output_path's suffix."""
        output_path = tmp_path / "segment.flac"

        result_path = AudioSeek.read_segment_to_file(
            test_wav_file, 0.0, 0.2, output_path=output_path
        )

        with sf.SoundFile(result_path) as f:
            assert f.format == "FLAC"

    def test_read_segment_to_file_explicit_format_overrides_extension(
        self, test_wav_file: Path, tmp_path: Path
    ) -> None:
        """format= takes precedence over output_path's extension."""
        output_path = tmp_path / "segment.wav"

        result_path = AudioSeek.read_segment_to_file(
            test_wav_file, 0.0, 0.2, output_path=output_path, format="AIFF"
        )

        with sf.SoundFile(result_path) as f:
            assert f.format == "AIFF"

    def test_read_segment_to_file_explicit_subtype(
        self, test_wav_file: Path, tmp_path: Path
    ) -> None:
        """subtype= controls the bit-depth/encoding metadata."""
        output_path = tmp_path / "segment_24bit.wav"

        result_path = AudioSeek.read_segment_to_file(
            test_wav_file, 0.0, 0.2, output_path=output_path, subtype="PCM_24"
        )

        with sf.SoundFile(result_path) as f:
            assert f.subtype == "PCM_24"

    def test_read_segment_to_file_invalid_subtype_raises(
        self, test_wav_file: Path, tmp_path: Path
    ) -> None:
        """An unsupported format/subtype combo raises ValueError up front."""
        output_path = tmp_path / "segment.ogg"

        with pytest.raises(ValueError):
            AudioSeek.read_segment_to_file(
                test_wav_file, 0.0, 0.2, output_path=output_path, subtype="PCM_16"
            )

    def test_read_segment_to_file_invalid_format_raises(
        self, test_wav_file: Path, tmp_path: Path
    ) -> None:
        """An unrecognized format raises ValueError up front."""
        output_path = tmp_path / "segment.out"

        with pytest.raises(ValueError):
            AudioSeek.read_segment_to_file(
                test_wav_file, 0.0, 0.2, output_path=output_path, format="NOTAFORMAT"
            )

    def test_read_segment_to_file_resamples_when_sample_rate_given(
        self, test_wav_file: Path, tmp_path: Path
    ) -> None:
        """sample_rate= resamples the segment instead of just relabeling it."""
        output_path = tmp_path / "segment_8k.wav"
        duration_sec = 0.2
        target_sample_rate = 8000

        result_path = AudioSeek.read_segment_to_file(
            test_wav_file,
            0.0,
            duration_sec,
            output_path=output_path,
            sample_rate=target_sample_rate,
        )

        with sf.SoundFile(result_path) as f:
            assert f.samplerate == target_sample_rate
            expected_frames = int(round(duration_sec * target_sample_rate))
            assert abs(f.frames - expected_frames) <= 1
