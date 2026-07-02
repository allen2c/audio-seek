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
