# Audio Seek

[![PyPI version](https://img.shields.io/pypi/v/audio-seek.svg)](https://pypi.org/project/audio-seek/)
[![Python Version](https://img.shields.io/pypi/pyversions/audio-seek.svg)](https://pypi.org/project/audio-seek/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

A lightweight Python library for reading arbitrary time slices out of common audio files without loading them fully — with minimal dependencies (`numpy` + `soundfile`).

## Key Features

* **Zero Waste:** Reads and decodes only the requested time slice, not the whole file.
* **Usual Formats:** Supports WAV, AIFF, FLAC, OGG/Vorbis, and MP3 via `soundfile`/`libsndfile`.
* **O(1) Where Possible:** Seeking into uncompressed PCM (WAV/AIFF) is a direct sample-offset seek, O(1). Compressed formats (MP3/FLAC/OGG) are frame/entropy coded, so seeking there is best-effort at frame granularity, not sample-exact.
* **Complexity Introspection:** `AudioSeek.seek_complexity()` tells you which regime a given file falls into.
* **Minimal Dependencies:** Just `numpy` and `soundfile` — no bundled codecs, no JIT toolchains.

## Installation

```bash
pip install audio-seek
```

## Quick Start

### Read a Specific Audio Segment

```python
from audio_seek import read_audio_segment

# Read 5 seconds starting at 2 minutes
segment = read_audio_segment(
    file_path="long_audio.wav",
    start_sec=120.0,
    duration_sec=5.0,
)
# Returns a numpy float32 array of shape (sample_rate * duration_sec,)
```

### Get Duration Without Loading Audio

```python
from audio_seek import AudioSeek

# Only reads the header, doesn't decode audio data
duration = AudioSeek.get_duration("audio.mp3")
print(f"Duration: {duration:.2f}s")
```

### Check Seek Complexity

```python
from audio_seek import AudioSeek

AudioSeek.seek_complexity("audio.wav")  # "O(1)"  — uncompressed PCM
AudioSeek.seek_complexity("audio.mp3")  # "O(n)"  — frame/entropy coded
```

### Extract a Segment to a New WAV File

```python
from audio_seek import AudioSeek

AudioSeek.read_segment_to_file(
    file_path="input.mp3",
    start_sec=10.0,
    duration_sec=15.0,
    output_path="clip.wav",  # written as PCM_16 WAV
)
```

### Mix Multi-Channel Audio Down to Mono

```python
from audio_seek import ensure_mono

mono = ensure_mono(stereo_data, style="soundfile")  # (samples, channels) -> (samples,)
```

## How It Works

`audio-seek` wraps `soundfile`/`libsndfile` and seeks directly via `SoundFile.seek()`:

* For **uncompressed PCM** (WAV, AIFF), each sample occupies a fixed number of bytes, so seeking is a direct file-offset computation — true O(1), sample-accurate.
* For **compressed formats** (MP3, FLAC, OGG/Vorbis), audio is encoded in frames/blocks of varying or fixed sample counts, sometimes with decoder warm-up requirements. Seeking there lands on the nearest frame boundary and is not guaranteed sample-exact — closer to O(n) in the worst case.

## Use Cases

Ideal for applications requiring low-latency access to specific segments of long audio recordings:

* **Machine Learning:** Dataset slicing without loading entire files
* **Web Services:** Real-time audio segment delivery
* **Audio Processing:** Random access pipelines

## Requirements

* Python >= 3.11
* numpy
* soundfile

## Testing

```bash
pytest tests/ -v
```

## License

Apache License 2.0 - see LICENSE file for details.
