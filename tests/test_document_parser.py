import pytest
import tempfile
import os

from panda_agent.rag.document_parser import DocumentParser, DocumentParseError


class TestDocumentParser:
    """Tests for DocumentParser class."""

    def test_parse_valid_pdf(self):
        """Test parsing a valid PDF file."""
        result = DocumentParser.parse_pdf('tests/fixtures/test_catalog.pdf')
        
        assert result["metadata"]["filename"] == "test_catalog.pdf"
        assert result["metadata"]["total_pages"] == 1
        assert result["metadata"]["extracted_pages"] == 1
        assert len(result["pages"]) == 1
        assert "SuperWidget Pro" in result["pages"][0]["text"]
        assert "$299.99" in result["pages"][0]["text"]
        assert "MegaGadget 3000" in result["pages"][0]["text"]
        assert "$599.50" in result["pages"][0]["text"]

    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist raises DocumentParseError."""
        with pytest.raises(DocumentParseError) as exc_info:
            DocumentParser.parse_pdf('/path/that/does/not/exist.pdf')
        
        assert "File not found" in str(exc_info.value)

    def test_parse_unsupported_format(self):
        """Test parsing an unsupported file format raises DocumentParseError."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is a text file, not a PDF")
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentParseError) as exc_info:
                DocumentParser.parse_pdf(temp_path)
            
            assert "Unsupported format" in str(exc_info.value)
            assert ".txt" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_parse_corrupted_pdf(self):
        """Test parsing a corrupted/invalid PDF raises DocumentParseError."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"This is not a valid PDF file content")
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentParseError) as exc_info:
                DocumentParser.parse_pdf(temp_path)
            
            assert "Failed to parse PDF" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_parse_empty_pdf(self):
        """Test parsing an empty (valid but no content) PDF."""
        # Create a minimal valid PDF with no text content
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        # Don't add any text
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
        
        try:
            pdf.output(temp_path)
            result = DocumentParser.parse_pdf(temp_path)
            
            assert result["metadata"]["total_pages"] == 1
            assert result["metadata"]["extracted_pages"] == 0
            assert len(result["pages"]) == 0
        finally:
            os.unlink(temp_path)

    def test_parse_pdf_with_multiple_pages(self):
        """Test parsing a PDF with multiple pages."""
        from fpdf import FPDF
        pdf = FPDF()
        
        # Page 1
        pdf.add_page()
        pdf.set_font('Helvetica', size=12)
        pdf.cell(0, 10, 'Page 1 Content', new_x='LMARGIN', new_y='NEXT')
        
        # Page 2
        pdf.add_page()
        pdf.cell(0, 10, 'Page 2 Content', new_x='LMARGIN', new_y='NEXT')
        
        # Page 3
        pdf.add_page()
        pdf.cell(0, 10, 'Page 3 Content', new_x='LMARGIN', new_y='NEXT')
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
        
        try:
            pdf.output(temp_path)
            result = DocumentParser.parse_pdf(temp_path)
            
            assert result["metadata"]["total_pages"] == 3
            assert result["metadata"]["extracted_pages"] == 3
            assert len(result["pages"]) == 3
            assert result["pages"][0]["page_number"] == 1
            assert result["pages"][1]["page_number"] == 2
            assert result["pages"][2]["page_number"] == 3
            assert "Page 1 Content" in result["pages"][0]["text"]
            assert "Page 2 Content" in result["pages"][1]["text"]
            assert "Page 3 Content" in result["pages"][2]["text"]
        finally:
            os.unlink(temp_path)

    def test_supported_formats_class_attribute(self):
        """Test that SUPPORTED_FORMATS includes .pdf."""
        assert ".pdf" in DocumentParser.SUPPORTED_FORMATS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])