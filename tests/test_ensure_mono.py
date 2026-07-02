"""Tests for audio_seek.ensure_mono (audio_seek/ensure_mono.py)."""

import numpy as np
import pytest

from audio_seek import ensure_mono


class TestEnsureMono:
    """Test suite for the ensure_mono channel-mixing helper."""

    def test_1d_passthrough(self) -> None:
        """Already-mono (1D) data is returned unchanged."""
        data: np.ndarray = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = ensure_mono(data)

        assert result is data

    def test_librosa_style_averages_axis_0(self) -> None:
        """style='librosa' expects (channels, samples) and averages axis 0."""
        channels: np.ndarray = np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float32
        )

        result = ensure_mono(channels, style="librosa")

        assert result.shape == (3,)
        assert np.allclose(result, 0.0)

    def test_soundfile_style_averages_axis_1(self) -> None:
        """style='soundfile' expects (samples, channels) and averages axis 1."""
        channels: np.ndarray = np.array(
            [[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]], dtype=np.float32
        )

        result = ensure_mono(channels, style="soundfile")

        assert result.shape == (3,)
        assert np.allclose(result, 0.0)

    def test_auto_detect_more_samples_than_channels(self) -> None:
        """style=None infers (samples, channels) when axis 0 is longer."""
        # 4 samples, 2 channels -> soundfile-style layout
        channels: np.ndarray = np.array(
            [[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]], dtype=np.float32
        )

        result = ensure_mono(channels)

        assert result.shape == (4,)
        assert np.allclose(result, 0.0)

    def test_auto_detect_more_channels_than_samples(self) -> None:
        """style=None infers (channels, samples) when axis 1 is longer."""
        # 2 channels, 4 samples -> librosa-style layout
        channels: np.ndarray = np.array(
            [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]], dtype=np.float32
        )

        result = ensure_mono(channels)

        assert result.shape == (4,)
        assert np.allclose(result, 0.0)

    def test_invalid_style_raises(self) -> None:
        """An unsupported style value raises ValueError."""
        channels: np.ndarray = np.zeros((2, 4), dtype=np.float32)

        with pytest.raises(ValueError):
            ensure_mono(channels, style="unsupported")  # type: ignore[arg-type]

    def test_invalid_ndim_raises(self) -> None:
        """Data with more than 2 dimensions raises ValueError."""
        data: np.ndarray = np.zeros((2, 3, 4), dtype=np.float32)

        with pytest.raises(ValueError):
            ensure_mono(data)
