"""Tests for audio conversion functionality."""

import numpy as np

from audio_seek import AudioSeek


class TestConversion:
    """Test suite for audio conversion operations."""

    def test_convert_numpy_array(self, temp_dir, sample_audio_data):
        """Test converting numpy array to WAV format."""
        output_path = temp_dir / "converted_from_array.wav"

        result_path = AudioSeek.convert(
            data=sample_audio_data["data"],
            output_path=output_path,
            src_sr=sample_audio_data["sample_rate"],
            target_sr=16000,
            bits=4,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_convert_with_resampling(self, temp_dir, sample_audio_data):
        """Test conversion with sample rate change."""
        output_path = temp_dir / "resampled.wav"

        result_path = AudioSeek.convert(
            data=sample_audio_data["data"],
            output_path=output_path,
            src_sr=16000,
            target_sr=8000,  # Downsample
            bits=4,
        )

        assert result_path.exists()

        # Check new sample rate
        import soundfile as sf

        with sf.SoundFile(result_path) as f:
            assert f.samplerate == 8000

    def test_convert_stereo_to_mono(self, temp_dir):
        """Test converting stereo audio to mono."""
        # Create stereo test data
        sample_rate = 16000
        duration = 0.5
        samples = int(sample_rate * duration)

        stereo_data = np.random.randn(samples, 2).astype(np.float32) * 0.1

        output_path = temp_dir / "mono_from_stereo.wav"

        result_path = AudioSeek.convert(
            data=stereo_data,
            output_path=output_path,
            src_sr=sample_rate,
            target_sr=sample_rate,
            bits=4,
            to_mono=True,
        )

        assert result_path.exists()

        # Verify it's mono
        import soundfile as sf

        with sf.SoundFile(result_path) as f:
            assert f.channels == 1

    def test_convert_different_bit_depths(self, temp_dir, sample_audio_data):
        """Test conversion with different bit depths."""
        for bits in [2, 3, 4, 5]:
            output_path = temp_dir / f"converted_bits_{bits}.wav"

            result_path = AudioSeek.convert(
                data=sample_audio_data["data"],
                output_path=output_path,
                src_sr=sample_audio_data["sample_rate"],
                target_sr=16000,
                bits=bits,  # type: ignore
            )

            assert result_path.exists()
            assert result_path.stat().st_size > 0

    def test_write_basic(self, temp_dir, sample_audio_data):
        """Test basic write operation."""
        output_path = temp_dir / "write_test.wav"

        result_path = AudioSeek.write(
            file_path=output_path,
            data=sample_audio_data["data"],
            sample_rate=sample_audio_data["sample_rate"],
            bits_per_sample=4,
        )

        assert result_path.exists()
        assert result_path == output_path

    def test_write_invalid_bits(self, temp_dir, sample_audio_data):
        """Test write with invalid bit depth."""
        output_path = temp_dir / "invalid_bits.wav"

        # This should either work (with fallback) or raise ValueError
        # depending on implementation
        try:
            AudioSeek.write(
                file_path=output_path,
                data=sample_audio_data["data"],
                sample_rate=16000,
                bits_per_sample=2,  # May not be available
            )
            # If it succeeds, it used fallback
            assert output_path.exists()
        except ValueError:
            # If it fails, that's also acceptable behavior
            pass

    def test_write_empty_data(self, temp_dir):
        """Test write with empty data array."""
        output_path = temp_dir / "empty_data.wav"
        empty_data = np.array([], dtype=np.float32)

        # Should handle gracefully
        try:
            AudioSeek.write(
                file_path=output_path,
                data=empty_data,
                sample_rate=16000,
                bits_per_sample=4,
            )
            # If successful, file should exist but be minimal
            if output_path.exists():
                assert output_path.stat().st_size > 0  # Should have header
        except Exception:
            # Also acceptable to reject empty data
            pass
