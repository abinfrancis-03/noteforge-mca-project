import os
from typing import Optional
from .base import (
    BaseDocumentExtractor,
    NormalizedDocument,
    NormalizedSection,
    EmptyDocumentException,
    DocumentExtractionException,
)


class PDFExtractor(BaseDocumentExtractor):
    """
    Format-specific extractor for PDF documents using PyMuPDF (fitz).
    Maintains page structure, headings, lists, and tables.
    """

    def extract(self, file_path_or_file, document_title: Optional[str] = None) -> NormalizedDocument:
        # Determine actual file path
        if hasattr(file_path_or_file, 'path'):
            file_path = file_path_or_file.path
        elif isinstance(file_path_or_file, str):
            file_path = file_path_or_file
        else:
            raise DocumentExtractionException("Invalid file representation provided for PDF extraction.")

        if not os.path.exists(file_path):
            raise DocumentExtractionException(f"File not found at path: {file_path}")

        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError:
                raise DocumentExtractionException("PyMuPDF (fitz) is required for PDF extraction.")

        sections = []
        total_text_chars = 0

        try:
            doc = fitz.open(file_path)
            num_pages = len(doc)

            if num_pages == 0:
                raise EmptyDocumentException("The supplied PDF document has 0 pages.")

            doc_name = document_title or os.path.basename(file_path)

            for page_index in range(num_pages):
                page_num = page_index + 1
                page = doc[page_index]

                # Extract text using blocks to detect layout & headings
                text_blocks = page.get_text("blocks")
                page_paragraphs = []
                detected_title = ""

                for block in text_blocks:
                    # block: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) >= 5 and block[4].strip():
                        b_text = block[4].strip()
                        # Simple heading heuristic: short first line on page or all-caps short block
                        if not detected_title and len(b_text) < 80 and ("\n" not in b_text or len(b_text.splitlines()) <= 2):
                            detected_title = b_text.splitlines()[0].strip()
                        page_paragraphs.append(b_text)

                page_text = "\n\n".join(page_paragraphs).strip()
                if not page_text:
                    # Fallback to standard get_text
                    page_text = page.get_text("text").strip()

                total_text_chars += len(page_text)

                section_title = detected_title if detected_title else f"Page {page_num}"

                sections.append(
                    NormalizedSection(
                        title=section_title,
                        content=page_text,
                        source_reference={"type": "page", "number": page_num},
                        metadata={"page_number": page_num, "has_text": bool(page_text)},
                    )
                )

            doc.close()

            # Check for scanned or image-only documents
            if total_text_chars < 50:
                # Attempt OCR fallback if pytesseract / PIL available
                ocr_success = False
                ocr_sections = []
                try:
                    import pytesseract
                    from PIL import Image
                    import io

                    doc_ocr = fitz.open(file_path)
                    ocr_total_chars = 0
                    for page_idx in range(min(len(doc_ocr), 15)): # Check first up to 15 pages
                        p = doc_ocr[page_idx]
                        pix = p.get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        txt = pytesseract.image_to_string(img).strip()
                        if txt:
                            ocr_total_chars += len(txt)
                            ocr_sections.append(
                                NormalizedSection(
                                    title=f"Page {page_idx + 1} (OCR)",
                                    content=txt,
                                    source_reference={"type": "page", "number": page_idx + 1},
                                    metadata={"page_number": page_idx + 1, "is_ocr": True},
                                )
                            )
                    doc_ocr.close()
                    if ocr_total_chars >= 50:
                        sections = ocr_sections
                        total_text_chars = ocr_total_chars
                        ocr_success = True
                except Exception:
                    ocr_success = False

                if not ocr_success:
                    raise EmptyDocumentException(
                        "This document appears to be scanned or contains no extractable text. "
                        "Please upload a document with searchable text for AI summarization."
                    )

            return NormalizedDocument(
                document_title=doc_name,
                source_type="pdf",
                sections=sections,
                total_units=num_pages,
            )

        except EmptyDocumentException:
            raise
        except Exception as e:
            raise DocumentExtractionException(f"Error processing PDF document: {str(e)}")
