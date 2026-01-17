"""Shared pytest fixtures for audio-seek library tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def module_version():
    """Returns the library version."""
    from audio_seek import __version__

    return __version__


@pytest.fixture(scope="session")
def temp_dir():
    """Creates a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_audio_data():
    """Generates sample audio data for testing (1 second, 16kHz, mono)."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    data = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

    return {"data": data, "sample_rate": sample_rate, "duration": duration}


@pytest.fixture(scope="session")
def test_wav_file(temp_dir, sample_audio_data):
    """Creates a test WAV file using AudioSeek."""
    from audio_seek import AudioSeek

    output_path = temp_dir / "test_sample.wav"

    AudioSeek.write(
        file_path=output_path,
        data=sample_audio_data["data"],
        sample_rate=sample_audio_data["sample_rate"],
        bits_per_sample=4,
    )

    return output_path
