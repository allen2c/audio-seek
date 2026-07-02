"""Shared pytest fixtures for audio-seek library tests."""

import base64
import tempfile
from pathlib import Path
from typing import Iterator, TypedDict

import numpy as np
import pytest
import soundfile as sf

DATA_DIR = Path(__file__).parent / "data"


class SampleAudioData(TypedDict):
    """Synthetic sine-wave audio data used across tests."""

    data: np.ndarray
    sample_rate: int
    duration: float


@pytest.fixture(scope="module")
def module_version() -> str:
    """Returns the library version."""
    from audio_seek import __version__

    return __version__


@pytest.fixture(scope="session")
def temp_dir() -> Iterator[Path]:
    """Creates a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_audio_data() -> SampleAudioData:
    """Generates sample audio data for testing (1 second, 16kHz, mono)."""
    sample_rate: int = 16000
    duration: float = 1.0
    frequency: float = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    data: np.ndarray = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

    return SampleAudioData(data=data, sample_rate=sample_rate, duration=duration)


@pytest.fixture(scope="session")
def test_wav_file(temp_dir: Path, sample_audio_data: SampleAudioData) -> Path:
    """Creates a test PCM WAV file."""
    output_path = temp_dir / "test_sample.wav"

    sf.write(
        output_path,
        sample_audio_data["data"],
        sample_audio_data["sample_rate"],
        format="WAV",
        subtype="PCM_16",
    )

    return output_path


@pytest.fixture(scope="session")
def sample_clip_bytes() -> bytes:
    """
    Decodes the base64-encoded 15s mp3 clip fixture (a random peek into
    a real-world field recording) back into raw mp3 bytes.
    """
    b64_text = (DATA_DIR / "sample_clip").read_text()
    return base64.b64decode(b64_text)


@pytest.fixture()
def sample_clip_file(tmp_path: Path, sample_clip_bytes: bytes) -> Path:
    """Writes the mp3 clip bytes to a suffix-less tmp file."""
    tmp_file = tmp_path / "sample_clip"
    tmp_file.write_bytes(sample_clip_bytes)
    return tmp_file
