"""Tests for audio_seek.utils (audio_seek/utils.py)."""

from pathlib import Path

import numpy as np
import pytest
from conftest import SampleAudioData

from audio_seek.utils import random_peek_segment


class TestRandomPeekSegment:
    """Test suite for the random_peek_segment utility."""

    def test_not_exported_at_top_level(self) -> None:
        """random_peek_segment stays out of the public audio_seek namespace."""
        import audio_seek

        assert not hasattr(audio_seek, "random_peek_segment")
        assert "random_peek_segment" not in audio_seek.__all__

    def test_duration_sec_must_be_positive(self, test_wav_file: Path) -> None:
        """duration_sec <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            random_peek_segment(test_wav_file, 0.0)

        with pytest.raises(ValueError):
            random_peek_segment(test_wav_file, -1.0)

    def test_nonexistent_file_raises(self) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            random_peek_segment("nonexistent_file.wav", 1.0)

    def test_shorter_than_file_stays_in_bounds(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """A segment shorter than the file never runs past end-of-file."""
        duration_sec = 0.3

        start_sec, data = random_peek_segment(test_wav_file, duration_sec, seed=42)

        total_duration = sample_audio_data["duration"]
        assert 0.0 <= start_sec <= total_duration - duration_sec
        expected_samples = int(duration_sec * sample_audio_data["sample_rate"])
        assert len(data) == expected_samples
        assert data.dtype == np.float32

    def test_longer_than_file_returns_whole_file(
        self, test_wav_file: Path, sample_audio_data: SampleAudioData
    ) -> None:
        """Requesting more than the file's duration returns the entire file."""
        start_sec, data = random_peek_segment(test_wav_file, 999.0, seed=1)

        assert start_sec == 0.0
        expected_samples = int(
            sample_audio_data["duration"] * sample_audio_data["sample_rate"]
        )
        assert len(data) == expected_samples

    def test_seed_is_reproducible(self, test_wav_file: Path) -> None:
        """The same seed always picks the same start point."""
        start_sec_1, _ = random_peek_segment(test_wav_file, 0.2, seed=7)
        start_sec_2, _ = random_peek_segment(test_wav_file, 0.2, seed=7)

        assert start_sec_1 == start_sec_2

    def test_works_on_mp3(self, sample_clip_file: Path) -> None:
        """random_peek_segment also works against a real-world mp3 clip."""
        duration_sec = 2.0

        start_sec, data = random_peek_segment(sample_clip_file, duration_sec, seed=3)

        assert start_sec >= 0.0
        assert data.dtype == np.float32
        assert len(data) > 0
