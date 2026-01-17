"""Tests for input/output operations."""

import numpy as np
import pytest

from audio_seek import AudioSeek, read_audio_segment


class TestIO:
    """Test suite for file I/O operations."""

    def test_write_read_cycle(self, temp_dir):
        """Test writing and reading back the same data."""
        # Create test data
        sample_rate = 16000
        duration = 0.5
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        original_data = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

        output_path = temp_dir / "write_read_test.wav"

        # Write
        AudioSeek.write(
            file_path=output_path,
            data=original_data,
            sample_rate=sample_rate,
            bits_per_sample=4,
        )

        # Read back
        read_data = read_audio_segment(
            file_path=output_path, start_sec=0.0, duration_sec=duration
        )

        # Compare
        assert len(read_data) == len(original_data)
        mse = np.mean((read_data - original_data) ** 2)
        assert mse < 0.01  # ADPCM compression tolerance

    def test_partial_read(self, test_wav_file, sample_audio_data):
        """Test reading only part of a file."""
        full_duration = sample_audio_data["duration"]
        partial_duration = full_duration / 2

        partial_data = read_audio_segment(
            file_path=test_wav_file, start_sec=0.0, duration_sec=partial_duration
        )

        expected_samples = int(partial_duration * sample_audio_data["sample_rate"])
        assert len(partial_data) == expected_samples

    def test_read_nonexistent_file(self):
        """Test reading a non-existent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            read_audio_segment(
                file_path="nonexistent.wav", start_sec=0.0, duration_sec=1.0
            )

    def test_get_duration_fast(self, test_wav_file):
        """Test that get_duration is fast (doesn't load audio data)."""
        import time

        # Should be very fast (< 100ms) even for large files
        start = time.time()
        duration = AudioSeek.get_duration(test_wav_file)
        elapsed = time.time() - start

        assert duration > 0
        assert elapsed < 0.1  # Should be instant

    def test_write_supports_path_types(self, temp_dir, sample_audio_data):
        """Test that write accepts both str and Path objects."""
        # Test with Path
        path_obj = temp_dir / "path_object.wav"
        result1 = AudioSeek.write(
            file_path=path_obj,
            data=sample_audio_data["data"],
            sample_rate=16000,
            bits_per_sample=4,
        )
        assert result1.exists()

        # Test with str
        str_path = str(temp_dir / "str_path.wav")
        result2 = AudioSeek.write(
            file_path=str_path,
            data=sample_audio_data["data"],
            sample_rate=16000,
            bits_per_sample=4,
        )
        assert result2.exists()

    def test_data_type_preservation(self, temp_dir):
        """Test that float32 data type is preserved through write/read cycle."""
        sample_rate = 16000
        duration = 0.2
        data = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        output_path = temp_dir / "dtype_test.wav"

        AudioSeek.write(
            file_path=output_path,
            data=data,
            sample_rate=sample_rate,
            bits_per_sample=4,
        )

        read_data = read_audio_segment(output_path, 0.0, duration)

        assert read_data.dtype == np.float32

    def test_value_range_preservation(self, temp_dir):
        """Test that audio value range is reasonably preserved."""
        sample_rate = 16000
        duration = 0.2

        # Create data with known range [-0.5, 0.5]
        data = np.random.uniform(-0.5, 0.5, int(sample_rate * duration)).astype(
            np.float32
        )

        output_path = temp_dir / "range_test.wav"

        AudioSeek.write(
            file_path=output_path,
            data=data,
            sample_rate=sample_rate,
            bits_per_sample=4,
        )

        read_data = read_audio_segment(output_path, 0.0, duration)

        # Values should be roughly in the same range (allowing for compression)
        assert read_data.min() >= -1.0
        assert read_data.max() <= 1.0
        assert read_data.min() < -0.2  # Should have negative values
        assert read_data.max() > 0.2  # Should have positive values
