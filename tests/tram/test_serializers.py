"""
Tests for PDF and Image serializers.
Uses synthetic minimal inputs — no real report files required.
"""
import pytest
from unittest.mock import patch, MagicMock
from tram.serializers.pdf_serializer import PDFSerializer
from tram.serializers.image_serializer import ImageSerializer


class TestPDFSerializer:

    def test_table_to_sentences_with_headers(self):
        """Table rows should be converted to header: value sentences."""
        s = PDFSerializer(b"", "test.pdf")
        table = [
            ["Technique", "ID", "Command"],
            ["PowerShell", "T1059.001", "powershell -enc abc"],
            ["WMI", "T1047", "wmic process call create"],
        ]
        result = s._table_to_sentences(table)
        assert "Technique: PowerShell" in result
        assert "ID: T1059.001" in result
        assert "T1047" in result

    def test_clean_text_removes_short_lines(self):
        s = PDFSerializer(b"", "test.pdf")
        text = "1\n\nThis is a valid sentence about lateral movement.\n2\nAnother valid line about credential dumping."
        result = s._clean_text(text)
        assert "1\n" not in result
        assert "lateral movement" in result

    def test_table_to_sentences_empty_rows_skipped(self):
        s = PDFSerializer(b"", "test.pdf")
        table = [["Header1", "Header2"], [None, None], ["val1", "val2"]]
        result = s._table_to_sentences(table)
        assert "val1" in result
        assert result.count("\n") == 0  # only one data row


class TestImageSerializer:

    def test_clean_text_removes_noise(self):
        s = ImageSerializer(b"", "test.png")
        text = "123\n\nThe adversary used PowerShell to execute commands.\n---\nAnother technique involved credential dumping via Mimikatz."
        result = s._clean_text(text)
        assert "PowerShell" in result
        assert "123" not in result

    def test_clean_text_deduplicates_consecutive_lines(self):
        s = ImageSerializer(b"", "test.png")
        text = "The attacker executed a script.\nThe attacker executed a script.\nLateral movement followed."
        result = s._clean_text(text)
        assert result.count("The attacker executed a script.") == 1

    def test_unsupported_mode_raises(self):
        """Non-image bytes should raise ValueError on open."""
        s = ImageSerializer(b"not an image", "test.png")
        with pytest.raises(ValueError, match="Cannot open image"):
            s.serialize()
