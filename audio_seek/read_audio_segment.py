from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio_segment(
    file_path: Path | str, start_sec: float, duration_sec: float
) -> np.ndarray:
    """
    General reader: supports PCM WAV, G.726 WAV (32k/48k/64k...), FLAC, etc.
    If the soundfile library recognizes the header, it can seek accurately.
    """
    try:
        with sf.SoundFile(file_path) as f:
            sr = f.samplerate  # Auto-detected, e.g., 16000

            # Compute frame (sample) positions
            start_frame = int(start_sec * sr)
            frames_to_read = int(duration_sec * sr)

            # Check total length to avoid out-of-bounds reads (optional but safe)
            if start_frame >= f.frames:
                return np.array([], dtype=np.float32)

            # Seek to the specified sample (O(1), very fast)
            f.seek(start_frame)

            # Read and decode automatically
            data = f.read(frames_to_read, dtype="float32")

            return data

    except Exception as e:
        raise ValueError(f"Read error: {e}")
