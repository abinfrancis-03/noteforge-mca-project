import os
from typing import Optional, List
from .base import (
    BaseDocumentExtractor,
    NormalizedDocument,
    NormalizedSection,
    EmptyDocumentException,
    DocumentExtractionException,
)


class PPTXExtractor(BaseDocumentExtractor):
    """
    Format-specific extractor for PowerPoint presentations (.pptx) using python-pptx.
    Extracts slide numbers, slide titles, bullet points, speaker notes, and tables.
    Includes automatic OpenXML fallback if python-pptx is unavailable.
    """

    def extract(self, file_path_or_file, document_title: Optional[str] = None) -> NormalizedDocument:
        if hasattr(file_path_or_file, 'path'):
            file_path = file_path_or_file.path
        elif isinstance(file_path_or_file, str):
            file_path = file_path_or_file
        else:
            raise DocumentExtractionException("Invalid file representation provided for PPTX extraction.")

        if not os.path.exists(file_path):
            raise DocumentExtractionException(f"File not found at path: {file_path}")

        # Try python-pptx first, fallback to native OpenXML zip extraction if needed
        prs = None
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
        except Exception:
            prs = None

        if prs is not None:
            return self._extract_via_python_pptx(prs, file_path, document_title)
        else:
            return self._extract_via_openxml(file_path, document_title)

    def _extract_via_python_pptx(self, prs, file_path: str, document_title: Optional[str]) -> NormalizedDocument:
        try:
            doc_name = document_title or os.path.basename(file_path)

            sections: List[NormalizedSection] = []
            total_text_chars = 0
            num_slides = len(prs.slides)

            if num_slides == 0:
                raise EmptyDocumentException("The PowerPoint presentation contains 0 slides.")

            for slide_idx, slide in enumerate(prs.slides):
                slide_num = slide_idx + 1
                slide_title = ""
                slide_items: List[str] = []

                # 1. Check title shape
                if slide.shapes.title and slide.shapes.title.text.strip():
                    slide_title = slide.shapes.title.text.strip()

                # 2. Iterate through shapes
                for shape in slide.shapes:
                    if shape == slide.shapes.title:
                        continue

                    # Text boxes & autoshapes
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            p_text = paragraph.text.strip()
                            if p_text:
                                if not slide_title and len(p_text) < 70 and not slide_items:
                                    slide_title = p_text
                                else:
                                    prefix = "  " * paragraph.level + "- " if paragraph.level > 0 else "- "
                                    slide_items.append(f"{prefix}{p_text}")
                                total_text_chars += len(p_text)

                    # Tables
                    if shape.has_table:
                        table = shape.table
                        table_lines = []
                        for r_idx, row in enumerate(table.rows):
                            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                            if r_idx == 0:
                                table_lines.append("| " + " | ".join(cells) + " |")
                                table_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                            else:
                                table_lines.append("| " + " | ".join(cells) + " |")
                        if table_lines:
                            t_str = "\n".join(table_lines)
                            total_text_chars += len(t_str)
                            slide_items.append(t_str)

                # 3. Speaker notes
                try:
                    if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                        notes_txt = slide.notes_slide.notes_text_frame.text.strip()
                        if notes_txt:
                            slide_items.append(f"\n*Speaker Notes:* {notes_txt}")
                            total_text_chars += len(notes_txt)
                except Exception:
                    pass

                content_str = "\n".join(slide_items).strip()
                title_str = slide_title if slide_title else f"Slide {slide_num}"

                sections.append(
                    NormalizedSection(
                        title=title_str,
                        content=content_str,
                        source_reference={"type": "slide", "number": slide_num},
                        metadata={"slide_number": slide_num, "has_content": bool(content_str)},
                    )
                )

            if total_text_chars < 30:
                raise EmptyDocumentException(
                    "This PowerPoint presentation appears to contain no extractable text."
                )

            return NormalizedDocument(
                document_title=doc_name,
                source_type="pptx",
                sections=sections,
                total_units=num_slides,
            )

        except EmptyDocumentException:
            raise
        except Exception as e:
            if file_path.lower().endswith('.ppt'):
                raise DocumentExtractionException(
                    "Legacy binary .ppt format cannot be parsed directly. "
                    "Please save or export as modern .pptx or .pdf format."
                )
            raise DocumentExtractionException(f"Error processing PPTX presentation: {str(e)}")

    def _extract_via_openxml(self, file_path: str, document_title: Optional[str]) -> NormalizedDocument:
        import zipfile
        import xml.etree.ElementTree as ET

        doc_name = document_title or os.path.basename(file_path)
        sections: List[NormalizedSection] = []
        total_text_chars = 0

        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                slide_entries = [
                    f for f in z.namelist()
                    if f.startswith('ppt/slides/slide') and f.endswith('.xml')
                ]

                def get_slide_num(name):
                    digits = ''.join(filter(str.isdigit, os.path.basename(name)))
                    return int(digits) if digits else 0

                slide_entries.sort(key=get_slide_num)

                if not slide_entries:
                    raise EmptyDocumentException("The PowerPoint presentation contains no slides.")

                for idx, entry in enumerate(slide_entries):
                    slide_num = idx + 1
                    xml_content = z.read(entry)
                    root = ET.fromstring(xml_content)

                    slide_title = ""
                    paragraphs = []

                    for p in root.iter():
                        if p.tag.endswith('}p'):
                            text_runs = []
                            for r in p.iter():
                                if r.tag.endswith('}t') and r.text:
                                    text_runs.append(r.text)
                            p_text = ''.join(text_runs).strip()
                            if p_text:
                                if not slide_title and len(p_text) < 70 and not paragraphs:
                                    slide_title = p_text
                                else:
                                    paragraphs.append(f"- {p_text}")
                                total_text_chars += len(p_text)

                    content_str = "\n".join(paragraphs).strip()
                    title_str = slide_title if slide_title else f"Slide {slide_num}"

                    sections.append(
                        NormalizedSection(
                            title=title_str,
                            content=content_str,
                            source_reference={"type": "slide", "number": slide_num},
                            metadata={"slide_number": slide_num, "has_content": bool(content_str)},
                        )
                    )

            if total_text_chars < 30:
                raise EmptyDocumentException(
                    "This PowerPoint presentation appears to contain no extractable text."
                )

            return NormalizedDocument(
                document_title=doc_name,
                source_type="pptx",
                sections=sections,
                total_units=len(slide_entries),
            )
        except EmptyDocumentException:
            raise
        except Exception as e:
            if file_path.lower().endswith('.ppt'):
                raise DocumentExtractionException(
                    "Legacy binary .ppt format cannot be parsed directly. "
                    "Please save or export as modern .pptx or .pdf format."
                )
            raise DocumentExtractionException(f"Error parsing PPTX via OpenXML: {str(e)}")
