from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class UnsupportedFileFormatException(Exception):
    """Raised when an uploaded document format is not supported."""
    pass


class EmptyDocumentException(Exception):
    """Raised when a document has no extractable text or appears to be a blank/un-OCR'd scan."""
    pass


class DocumentExtractionException(Exception):
    """Raised when an unexpected error occurs during file parsing."""
    pass


@dataclass
class NormalizedSection:
    title: str = ""
    content: str = ""
    source_reference: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "source_reference": self.source_reference,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedDocument:
    document_title: str = ""
    source_type: str = ""
    sections: List[NormalizedSection] = field(default_factory=list)
    total_units: int = 0
    raw_text: str = ""

    def get_combined_text(self) -> str:
        if self.raw_text:
            return self.raw_text
        parts = []
        for sec in self.sections:
            header = f"### {sec.title}" if sec.title else ""
            ref = ""
            if sec.source_reference:
                ref_type = sec.source_reference.get("type", "unit")
                ref_num = sec.source_reference.get("number", "")
                if ref_num:
                    ref = f"[{ref_type.capitalize()} {ref_num}]"
            
            section_header = f"{header} {ref}".strip()
            if section_header:
                parts.append(section_header)
            if sec.content:
                parts.append(sec.content)
        self.raw_text = "\n\n".join(parts)
        return self.raw_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_title": self.document_title,
            "source_type": self.source_type,
            "total_units": self.total_units,
            "sections": [s.to_dict() for s in self.sections],
            "text_length": len(self.get_combined_text()),
        }


class BaseDocumentExtractor(ABC):
    """Abstract base class for all format-specific extractors."""

    @abstractmethod
    def extract(self, file_path_or_file, document_title: Optional[str] = None) -> NormalizedDocument:
        """
        Parses document file and returns a unified NormalizedDocument.
        """
        pass
