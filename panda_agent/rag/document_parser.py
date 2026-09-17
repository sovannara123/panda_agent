from pathlib import Path
from PyPDF2 import PdfReader
from panda_agent.core.logger import log_event


class DocumentParseError(Exception):
    pass


class DocumentParser:
    """Parse documents and extract text with metadata."""
    
    SUPPORTED_FORMATS = {".pdf"}
    
    @classmethod
    def parse_pdf(cls, file_path: str) -> dict:
        """Extract text and metadata from a PDF file."""
        
        path = Path(file_path)
        
        if not path.exists():
            raise DocumentParseError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            raise DocumentParseError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {cls.SUPPORTED_FORMATS}"
            )
        
        try:
            reader = PdfReader(file_path)
            
            # Extract metadata
            metadata = {
                "filename": path.name,
                "total_pages": len(reader.pages),
                "format": "pdf"
            }
            
            # Extract text page by page
            pages = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                
                # Clean up whitespace
                text = " ".join(text.split())
                
                if text.strip():
                    pages.append({
                        "page_number": page_num,
                        "text": text
                    })
            
            metadata["extracted_pages"] = len(pages)
            
            log_event("document_parsed", {
                "filename": metadata["filename"],
                "total_pages": metadata["total_pages"],
                "extracted_pages": metadata["extracted_pages"]
            })
            
            return {
                "metadata": metadata,
                "pages": pages
            }
            
        except Exception as error:
            log_event("document_parse_error", {
                "filename": path.name,
                "error": str(error)
            })
            raise DocumentParseError(f"Failed to parse PDF: {str(error)}")