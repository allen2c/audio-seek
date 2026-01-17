# Audio Seek

[![PyPI version](https://img.shields.io/pypi/v/audio-seek.svg)](https://pypi.org/project/audio-seek/)
[![Python Version](https://img.shields.io/pypi/pyversions/audio-seek.svg)](https://pypi.org/project/audio-seek/)
[![License](https://img.shields.io/pypi/l/audio-seek.svg)](https://opensource.org/licenses/MIT)

A lightweight Python library designed for **precision seeking** and **zero-waste decoding** of compressed audio files with maximum cross-platform compatibility.

## Key Features

* **O(1) Seeking:** Sample-accurate seeking without parsing frame headers or processing bit reservoirs.
* **Zero Waste:** Reads and decodes only the requested time slice. No "warm-up" samples or overlapping frames required.
* **Smart Format Selection:** Automatically selects the best available seekable compression format based on your system.
* **Cross-Platform Compatibility:** Tests seekability at runtime and falls back gracefully to ensure reliability.
* **High Efficiency:** Wraps `libsndfile` (via `pysoundfile`) for C-level performance.

## Installation

```bash
pip install audio-seek
```

## Quick Start

### Convert Audio to Seekable Format

```python
from audio_seek import AudioSeek

# Convert MP3 to seekable WAV format
AudioSeek.convert_from_file(
    input_path="input.mp3",
    output_path="output.wav",
    target_sr=16000,
    bits=4,  # Auto-selects best seekable format
)
```

### Read Specific Audio Segments (O(1) Seeking)

```python
from audio_seek import read_audio_segment

# Read 5 seconds starting at 2 minutes
segment = read_audio_segment(
    file_path="long_audio.wav",
    start_sec=120.0,
    duration_sec=5.0,
)
# Returns numpy array of shape (sample_rate * duration,)
```

### Get Duration Without Loading Audio

```python
from audio_seek import AudioSeek

# O(1) operation - only reads header
duration = AudioSeek.get_duration("audio.wav")
print(f"Duration: {duration:.2f}s")
```

### Write Numpy Array to Seekable Format

```python
import numpy as np
from audio_seek import AudioSeek

# Create audio data
audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)

# Write to seekable WAV
AudioSeek.write(
    file_path="output.wav",
    data=audio_data,
    sample_rate=16000,
    bits_per_sample=4,
)
```

## How It Works

Unlike formats like MP3 or AAC that require sequential decoding, `audio-seek` uses compression formats that support true random access:

1. **Runtime Format Detection:** Tests available formats for seekability on your system
2. **Smart Fallback:** Automatically selects IMA_ADPCM or MS_ADPCM if ideal formats aren't available
3. **Guaranteed Seekability:** Only uses formats that pass actual seek tests

## Use Cases

Ideal for applications requiring low-latency access to specific segments of long audio recordings:

* **Machine Learning:** Dataset slicing without loading entire files
* **Web Services:** Real-time audio segment delivery
* **Telephony:** Archive retrieval and call analysis
* **Audio Processing:** Random access pipelines

## Requirements

* Python >= 3.11
* numpy
* soundfile
* librosa (for format conversion)

## Testing

```bash
pytest tests/ -v
```

## License

MIT License - see LICENSE file for details.
