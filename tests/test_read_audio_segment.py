"""Tests for audio_seek.read_audio_segment (audio_seek/read_audio_segment.py)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import SampleAudioData

from audio_seek import read_audio_segment


class TestReadAudioSegment:
    """Test suite for the standalone read_audio_segment function."""

    def test_read_segment_beginning(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Test seeking to the beginning of the file."""
        start_sec: float = 0.0
        duration_sec: float = 0.3

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        expected_samples = int(duration_sec * sample_audio_data["sample_rate"])
        assert len(segment) == expected_samples
        assert segment.dtype == np.float32

    def test_read_segment_middle(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Test seeking to the middle of the file."""
        start_sec: float = 0.5
        duration_sec: float = 0.2

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        expected_samples = int(duration_sec * sample_audio_data["sample_rate"])
        assert len(segment) == expected_samples

    def test_read_segment_near_end(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Test seeking near the end of the file."""
        start_sec: float = 0.8
        duration_sec: float = 0.15

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        expected_samples = int(duration_sec * sample_audio_data["sample_rate"])
        assert len(segment) == expected_samples

    def test_read_segment_beyond_end(self, test_wav_file: Path) -> None:
        """Test seeking beyond file length returns empty data."""
        start_sec: float = 10.0  # Beyond 1 second file
        duration_sec: float = 1.0

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        assert len(segment) == 0

    def test_read_segment_accuracy(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Test that seek+read produces sample-accurate results."""
        start_sec: float = 0.2
        duration_sec: float = 0.3

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        sr = sample_audio_data["sample_rate"]
        start_idx = int(start_sec * sr)
        end_idx = start_idx + int(duration_sec * sr)
        expected = sample_audio_data["data"][start_idx:end_idx]

        mse = np.mean((segment - expected) ** 2)
        assert mse < 0.001, f"Seek accuracy too low, MSE: {mse}"

    def test_multiple_seeks_same_file(self, test_wav_file: Path) -> None:
        """Test multiple sequential seeks on the same file."""
        positions: list[tuple[float, float]] = [
            (0.0, 0.1),
            (0.3, 0.1),
            (0.6, 0.1),
            (0.2, 0.1),
        ]

        for start, duration in positions:
            segment = read_audio_segment(test_wav_file, start, duration)
            assert len(segment) > 0

    def test_read_write_cycle(self, temp_dir: Path) -> None:
        """Test writing and reading back the same data."""
        sample_rate: int = 16000
        duration: float = 0.5
        frequency: float = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        original_data = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

        output_path = temp_dir / "write_read_test.wav"
        sf.write(
            output_path, original_data, sample_rate, format="WAV", subtype="PCM_16"
        )

        read_data = read_audio_segment(
            file_path=output_path, start_sec=0.0, duration_sec=duration
        )

        assert len(read_data) == len(original_data)
        mse = np.mean((read_data - original_data) ** 2)
        assert mse < 0.001  # PCM_16 quantization tolerance

    def test_read_nonexistent_file(self) -> None:
        """Test reading a non-existent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            read_audio_segment(
                file_path="nonexistent.wav", start_sec=0.0, duration_sec=1.0
            )

    def test_data_type_preservation(self, temp_dir: Path) -> None:
        """Test that float32 data type is preserved through write/read cycle."""
        sample_rate: int = 16000
        duration: float = 0.2
        data = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        output_path = temp_dir / "dtype_test.wav"
        sf.write(output_path, data, sample_rate, format="WAV", subtype="PCM_16")

        read_data = read_audio_segment(output_path, 0.0, duration)

        assert read_data.dtype == np.float32

    def test_value_range_preservation(self, temp_dir: Path) -> None:
        """Test that audio value range is reasonably preserved."""
        sample_rate: int = 16000
        duration: float = 0.2

        data = np.random.uniform(-0.5, 0.5, int(sample_rate * duration)).astype(
            np.float32
        )

        output_path = temp_dir / "range_test.wav"
        sf.write(output_path, data, sample_rate, format="WAV", subtype="PCM_16")

        read_data = read_audio_segment(output_path, 0.0, duration)

        assert read_data.min() >= -1.0
        assert read_data.max() <= 1.0
        assert read_data.min() < -0.2  # Should have negative values
        assert read_data.max() > 0.2  # Should have positive values

    def test_read_segment_from_mp3(self, sample_clip_file: Path) -> None:
        """Test reading from a real-world mp3 clip (no file extension)."""
        segment = read_audio_segment(sample_clip_file, start_sec=0.0, duration_sec=2.0)

        assert segment.dtype == np.float32
        assert len(segment) > 0
