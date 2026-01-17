from typing import Literal

import numpy as np


def ensure_mono(
    data: np.ndarray, *, style: Literal["librosa", "soundfile"] | None = None
) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging across channels.
    Supports librosa (channels, samples) and soundfile (samples, channels) formats.
    """
    if data.ndim == 1:
        return data

    elif data.ndim == 2:
        # data shape is either (samples, channels) or (channels, samples)
        # librosa typically produces (channels, samples),
        # soundfile uses (samples, channels)
        if style is None:
            if data.shape[1] < data.shape[0]:  # Determine which dimension is channels
                data = np.mean(data, axis=1)
            else:
                data = np.mean(data, axis=0)
        elif style == "librosa":
            data = np.mean(data, axis=0)
        elif style == "soundfile":
            data = np.mean(data, axis=1)
        else:
            raise ValueError(f"Unsupported style: {style}")

        return data

    else:
        raise ValueError(f"Unsupported number of dimensions: {data.ndim}")
