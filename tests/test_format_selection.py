"""Tests for automatic format selection and seekability detection."""

import warnings

from audio_seek import AudioSeek
from audio_seek._audio_seek import SUBTYPE_CACHE


class TestFormatSelection:
    """Test suite for format selection logic."""

    def test_resolve_subtype_returns_string(self):
        """Test that resolve_best_subtype returns a valid string."""
        result = AudioSeek.resolve_best_subtype(4)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_bit_depths_resolve(self):
        """Test that all supported bit depths can resolve to a format."""
        for bits in [2, 3, 4, 5]:
            result = AudioSeek.resolve_best_subtype(bits)  # type: ignore
            assert isinstance(result, str)
            assert result in ["IMA_ADPCM", "MS_ADPCM", "G726_32", "G721_32"]

    def test_cache_populated_after_resolve(self):
        """Test that cache is populated after resolving format."""
        # Clear cache first
        SUBTYPE_CACHE.clear()

        bits = 4
        AudioSeek.resolve_best_subtype(bits)

        assert bits in SUBTYPE_CACHE
        assert isinstance(SUBTYPE_CACHE[bits], dict)
        assert "subtype" in SUBTYPE_CACHE[bits]
        assert "seekable" in SUBTYPE_CACHE[bits]
        assert SUBTYPE_CACHE[bits]["seekable"] is True

    def test_cache_reused_on_second_call(self):
        """Test that cached result is reused on subsequent calls."""
        SUBTYPE_CACHE.clear()

        # First call
        AudioSeek.resolve_best_subtype(4)

        # Modify cache to verify it's being used
        original_value = SUBTYPE_CACHE[4]["subtype"]
        SUBTYPE_CACHE[4]["subtype"] = "CACHED_VALUE"

        # Second call should use cache
        result2 = AudioSeek.resolve_best_subtype(4)

        assert result2 == "CACHED_VALUE"

        # Restore original value
        SUBTYPE_CACHE[4]["subtype"] = original_value

    def test_warning_on_unsupported_bit_depth(self):
        """Test that warning is raised when falling back to compatible format."""
        SUBTYPE_CACHE.clear()

        # bits=2 likely doesn't have seekable format, should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AudioSeek.resolve_best_subtype(2)

            # Should have warned about fallback
            assert len(w) >= 1
            assert "falling back" in str(w[0].message).lower()
            assert "compatibility" in str(w[0].message).lower()

    def test_seekability_testing(self):
        """Test that test_seekability correctly identifies seekable formats."""
        # Test known seekable format
        is_seekable = AudioSeek.test_seekability("IMA_ADPCM")
        assert is_seekable is True

        # Test non-existent format (should return False)
        is_seekable = AudioSeek.test_seekability("NONEXISTENT_FORMAT_12345")
        assert is_seekable is False

    def test_subtype_info_structure(self):
        """Test that SubtypeInfo has correct structure."""
        SUBTYPE_CACHE.clear()

        AudioSeek.resolve_best_subtype(4)
        info = SUBTYPE_CACHE[4]

        assert "subtype" in info
        assert "seekable" in info
        assert "bits_per_sample" in info

        assert isinstance(info["subtype"], str)
        assert isinstance(info["seekable"], bool)
        assert isinstance(info["bits_per_sample"], int)
