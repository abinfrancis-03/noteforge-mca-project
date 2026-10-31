import os
from typing import Optional
from .base import (
    BaseDocumentExtractor,
    NormalizedDocument,
    NormalizedSection,
    UnsupportedFileFormatException,
    EmptyDocumentException,
    DocumentExtractionException,
)
from .pdf_extractor import PDFExtractor
from .docx_extractor import DOCXExtractor
from .pptx_extractor import PPTXExtractor


class TextFileExtractor(BaseDocumentExtractor):
    """Fallback extractor for plain text and markdown study files."""

    def extract(self, file_path_or_file, document_title: Optional[str] = None) -> NormalizedDocument:
        if hasattr(file_path_or_file, 'path'):
            file_path = file_path_or_file.path
        elif isinstance(file_path_or_file, str):
            file_path = file_path_or_file
        else:
            raise DocumentExtractionException("Invalid file representation provided.")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()

            if not content:
                raise EmptyDocumentException("Text document contains no readable characters.")

            doc_name = document_title or os.path.basename(file_path)
            # Split by markdown headers if available
            sections = []
            lines = content.splitlines()
            cur_heading = "Notes"
            cur_lines = []
            sec_idx = 1

            for line in lines:
                if line.startswith("#"):
                    if cur_lines:
                        sections.append(
                            NormalizedSection(
                                title=cur_heading,
                                content="\n".join(cur_lines).strip(),
                                source_reference={"type": "section", "number": sec_idx},
                            )
                        )
                        sec_idx += 1
                        cur_lines = []
                    cur_heading = line.lstrip("#").strip()
                else:
                    cur_lines.append(line)

            if cur_lines or not sections:
                sections.append(
                    NormalizedSection(
                        title=cur_heading,
                        content="\n".join(cur_lines).strip() or content,
                        source_reference={"type": "section", "number": sec_idx},
                    )
                )

            return NormalizedDocument(
                document_title=doc_name,
                source_type="text",
                sections=sections,
                total_units=len(sections),
            )
        except EmptyDocumentException:
            raise
        except Exception as e:
            raise DocumentExtractionException(f"Error reading text document: {str(e)}")


def get_extractor(file_path_or_extension: str) -> BaseDocumentExtractor:
    """
    Factory function to return the appropriate extractor for the given file extension.
    Raises UnsupportedFileFormatException if format is not supported.
    """
    ext = ""
    if hasattr(file_path_or_extension, 'path'):
        ext = os.path.splitext(file_path_or_extension.path)[1].lower()
    elif isinstance(file_path_or_extension, str):
        if "." in file_path_or_extension:
            ext = os.path.splitext(file_path_or_extension)[1].lower()
        else:
            ext = f".{file_path_or_extension.lower().lstrip('.')}"

    if ext == '.pdf':
        return PDFExtractor()
    elif ext == '.docx':
        return DOCXExtractor()
    elif ext == '.pptx':
        return PPTXExtractor()
    elif ext in ('.txt', '.md'):
        return TextFileExtractor()
    elif ext in ('.doc', '.ppt'):
        # Inform student of legacy formats
        raise UnsupportedFileFormatException(
            f"The legacy '{ext}' format is not directly supported for AI summarization. "
            f"Please save or export the file as modern '{ext}x' or PDF."
        )
    else:
        raise UnsupportedFileFormatException(
            f"The '{ext or 'unknown'}' file format is not currently supported for AI summarization. "
            "Supported educational formats include: PDF (.pdf), PowerPoint (.pptx), and Word (.docx)."
        )


def extract_document(file_path_or_file, document_title: Optional[str] = None) -> NormalizedDocument:
    """
    Convenience wrapper to extract any supported document through the factory.
    """
    extractor = get_extractor(file_path_or_file)
    return extractor.extract(file_path_or_file, document_title=document_title)
