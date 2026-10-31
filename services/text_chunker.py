from dataclasses import dataclass, field
from typing import List, Dict, Any
from .document_extractor.base import NormalizedDocument, NormalizedSection


@dataclass
class SmartChunk:
    chunk_index: int
    title: str
    unit_type: str
    start_unit: int
    end_unit: int
    sections: List[NormalizedSection] = field(default_factory=list)
    formatted_text: str = ""

    def get_text(self) -> str:
        if self.formatted_text:
            return self.formatted_text
        parts = []
        for s in self.sections:
            ref_str = ""
            if s.source_reference:
                rtype = s.source_reference.get("type", "unit")
                rnum = s.source_reference.get("number", "")
                if rnum:
                    ref_str = f"[{rtype.upper()} {rnum}]"
            
            header = f"### {s.title} {ref_str}".strip()
            if header:
                parts.append(header)
            if s.content:
                parts.append(s.content)
        self.formatted_text = "\n\n".join(parts)
        return self.formatted_text


class SmartChunker:
    """
    Intelligent document chunker that splits normalized documents along logical boundaries:
    - Pages for PDF
    - Slides for PPTX
    - Headings/sections for DOCX
    Preserves section hierarchy, headings, and source references.
    """

    def __init__(self, max_chars_per_chunk: int = 14000, single_pass_threshold: int = 18000):
        self.max_chars_per_chunk = max_chars_per_chunk
        self.single_pass_threshold = single_pass_threshold

    def is_single_pass(self, doc: NormalizedDocument) -> bool:
        """
        Determines whether document fits comfortably in a single LLM synthesis pass.
        """
        total_len = len(doc.get_combined_text())
        return total_len <= self.single_pass_threshold

    def chunk(self, doc: NormalizedDocument) -> List[SmartChunk]:
        """
        Chunks the normalized document along logical section boundaries.
        """
        if not doc.sections:
            return []

        chunks: List[SmartChunk] = []
        current_sections: List[NormalizedSection] = []
        current_len = 0
        chunk_idx = 1

        unit_type = "unit"
        if doc.source_type == "pdf":
            unit_type = "page"
        elif doc.source_type == "pptx":
            unit_type = "slide"
        elif doc.source_type == "docx":
            unit_type = "section"

        def create_chunk(sections: List[NormalizedSection], idx: int) -> SmartChunk:
            first_num = sections[0].source_reference.get("number", idx) if sections else idx
            last_num = sections[-1].source_reference.get("number", idx) if sections else idx
            
            # Title reflects unit range
            if first_num == last_num:
                chunk_title = f"{unit_type.capitalize()} {first_num}"
            else:
                chunk_title = f"{unit_type.capitalize()}s {first_num}–{last_num}"

            c = SmartChunk(
                chunk_index=idx,
                title=chunk_title,
                unit_type=unit_type,
                start_unit=first_num,
                end_unit=last_num,
                sections=list(sections),
            )
            c.get_text() # generate formatted text
            return c

        for sec in doc.sections:
            sec_len = len(sec.title) + len(sec.content) + 50
            if current_sections and (current_len + sec_len > self.max_chars_per_chunk):
                # Flush current chunk
                chunks.append(create_chunk(current_sections, chunk_idx))
                chunk_idx += 1
                current_sections = [sec]
                current_len = sec_len
            else:
                current_sections.append(sec)
                current_len += sec_len

        if current_sections:
            chunks.append(create_chunk(current_sections, chunk_idx))

        return chunks
