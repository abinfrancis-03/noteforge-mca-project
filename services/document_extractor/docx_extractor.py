import os
from typing import Optional, List
from .base import (
    BaseDocumentExtractor,
    NormalizedDocument,
    NormalizedSection,
    EmptyDocumentException,
    DocumentExtractionException,
)


class DOCXExtractor(BaseDocumentExtractor):
    """
    Format-specific extractor for DOCX / Word documents using python-docx.
    Maintains document order, headings, paragraphs, bullet points, and tables.
    """

    def extract(self, file_path_or_file, document_title: Optional[str] = None) -> NormalizedDocument:
        if hasattr(file_path_or_file, 'path'):
            file_path = file_path_or_file.path
        elif isinstance(file_path_or_file, str):
            file_path = file_path_or_file
        else:
            raise DocumentExtractionException("Invalid file representation provided for DOCX extraction.")

        if not os.path.exists(file_path):
            raise DocumentExtractionException(f"File not found at path: {file_path}")

        try:
            import docx
        except ImportError:
            raise DocumentExtractionException("python-docx is required for DOCX extraction.")

        try:
            doc = docx.Document(file_path)
            doc_name = document_title or os.path.basename(file_path)

            sections: List[NormalizedSection] = []
            current_heading = "Introduction"
            current_paragraphs: List[str] = []
            section_counter = 1
            total_text_chars = 0

            def commit_section():
                nonlocal section_counter, current_heading, current_paragraphs
                content = "\n\n".join(current_paragraphs).strip()
                if content:
                    sections.append(
                        NormalizedSection(
                            title=current_heading,
                            content=content,
                            source_reference={"type": "section", "number": section_counter},
                            metadata={"section_index": section_counter},
                        )
                    )
                    section_counter += 1
                current_paragraphs = []

            # Traverse paragraphs and tables in natural order if possible
            for p in doc.paragraphs:
                text = p.text.strip()
                if not text:
                    continue

                total_text_chars += len(text)
                style_name = (p.style.name or "").lower()

                # If paragraph is a heading or title, start a new section
                if "heading" in style_name or "title" in style_name:
                    if current_paragraphs:
                        commit_section()
                    current_heading = text
                else:
                    if "list" in style_name or "bullet" in style_name:
                        current_paragraphs.append(f"- {text}")
                    else:
                        current_paragraphs.append(text)

            # Also extract tables
            for table in doc.tables:
                table_lines = []
                for row_idx, row in enumerate(table.rows):
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    if row_idx == 0:
                        header_line = "| " + " | ".join(cells) + " |"
                        separator_line = "| " + " | ".join(["---"] * len(cells)) + " |"
                        table_lines.append(header_line)
                        table_lines.append(separator_line)
                    else:
                        table_lines.append("| " + " | ".join(cells) + " |")

                if table_lines:
                    t_str = "\n".join(table_lines)
                    total_text_chars += len(t_str)
                    current_paragraphs.append(t_str)

            # Commit final section
            commit_section()

            if total_text_chars < 30 or not sections:
                raise EmptyDocumentException(
                    "This DOCX document appears to contain no extractable text."
                )

            return NormalizedDocument(
                document_title=doc_name,
                source_type="docx",
                sections=sections,
                total_units=len(sections),
            )

        except EmptyDocumentException:
            raise
        except Exception as e:
            raise DocumentExtractionException(f"Error processing DOCX document: {str(e)}")
