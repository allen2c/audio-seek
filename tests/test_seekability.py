"""Tests for O(1) seeking functionality."""

import numpy as np
import pytest

from audio_seek import AudioSeek, read_audio_segment


class TestSeekability:
    """Test suite for audio seeking operations."""

    def test_read_segment_beginning(self, test_wav_file, sample_audio_data):
        """Test seeking to the beginning of the file."""
        start_sec = 0.0
        duration_sec = 0.3

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        expected_samples = int(duration_sec * sample_audio_data["sample_rate"])
        assert len(segment) == expected_samples
        assert segment.dtype == np.float32

    def test_read_segment_middle(self, test_wav_file, sample_audio_data):
        """Test seeking to the middle of the file."""
        start_sec = 0.5
        duration_sec = 0.2

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        expected_samples = int(duration_sec * sample_audio_data["sample_rate"])
        assert len(segment) == expected_samples

    def test_read_segment_near_end(self, test_wav_file, sample_audio_data):
        """Test seeking near the end of the file."""
        start_sec = 0.8
        duration_sec = 0.15

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        expected_samples = int(duration_sec * sample_audio_data["sample_rate"])
        assert len(segment) == expected_samples

    def test_read_segment_beyond_end(self, test_wav_file):
        """Test seeking beyond file length returns empty or truncated data."""
        start_sec = 10.0  # Beyond 1 second file
        duration_sec = 1.0

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        # Should return empty array
        assert len(segment) == 0

    def test_read_segment_accuracy(self, test_wav_file, sample_audio_data):
        """Test that seek+read produces accurate results."""
        start_sec = 0.2
        duration_sec = 0.3

        segment = read_audio_segment(test_wav_file, start_sec, duration_sec)

        # Calculate expected segment from original data
        sr = sample_audio_data["sample_rate"]
        start_idx = int(start_sec * sr)
        end_idx = start_idx + int(duration_sec * sr)
        expected = sample_audio_data["data"][start_idx:end_idx]

        # Compare with small tolerance (ADPCM compression)
        mse = np.mean((segment - expected) ** 2)
        assert mse < 0.01, f"Seek accuracy too low, MSE: {mse}"

    def test_multiple_seeks_same_file(self, test_wav_file):
        """Test multiple sequential seeks on the same file."""
        positions = [(0.0, 0.1), (0.3, 0.1), (0.6, 0.1), (0.2, 0.1)]

        for start, duration in positions:
            segment = read_audio_segment(test_wav_file, start, duration)
            assert len(segment) > 0

    def test_get_duration_accuracy(self, test_wav_file, sample_audio_data):
        """Test that get_duration returns accurate duration."""
        duration = AudioSeek.get_duration(test_wav_file)

        expected = sample_audio_data["duration"]
        # ADPCM formats may add padding, allow 50ms tolerance
        assert abs(duration - expected) < 0.05

    def test_get_duration_nonexistent_file(self):
        """Test that get_duration raises error for non-existent files."""
        with pytest.raises(FileNotFoundError):
            AudioSeek.get_duration("nonexistent_file.wav")
