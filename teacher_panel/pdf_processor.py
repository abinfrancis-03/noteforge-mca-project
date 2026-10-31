import os
import re
from collections import Counter
import pymupdf  # PyMuPDF
from accounts.models import QuizExtractedPage


def clean_page_whitespace(text: str) -> str:
    """
    Cleans unnecessary whitespace without destroying paragraph and line structures.
    - Strips null bytes and control characters
    - Fixes hyphenated line breaks (e.g. 'algo-\nrithm' -> 'algorithm')
    - Collapses multiple horizontal spaces
    - Collapses 3+ consecutive line breaks into 2 (clean paragraph separation)
    """
    if not text:
        return ""

    # Remove null characters and non-printable control chars except \n and \t
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Rejoin words broken across line breaks with a hyphen
    text = re.sub(r'(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)', r'\1\2', text)

    # Normalize horizontal whitespace (tabs, non-breaking spaces, multiple spaces)
    text = re.sub(r'[^\S\r\n]+', ' ', text)

    # Collapse excessive blank lines (more than 2 consecutive newlines to 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def detect_repeated_headers_footers(raw_pages: list) -> tuple:
    """
    Detects repeated header and footer lines that occur on >= 50% of pages.
    These typically represent document watermarks, static course headers, or page number stamps.
    """
    total = len(raw_pages)
    if total < 2:
        return set(), set()

    candidate_headers = []
    candidate_footers = []

    for text in raw_pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            # First 1-2 lines as header candidates (max 100 chars to avoid capturing long paragraph intros)
            for h in lines[:2]:
                if len(h) <= 100:
                    candidate_headers.append(h)
            # Last 1-2 lines as footer candidates
            for f in lines[-2:]:
                if len(f) <= 100:
                    candidate_footers.append(f)

    threshold = max(2, int(total * 0.45))
    common_headers = {line for line, count in Counter(candidate_headers).items() if count >= threshold}
    common_footers = {line for line, count in Counter(candidate_footers).items() if count >= threshold}

    return common_headers, common_footers


def clean_page_lines(text: str, common_headers: set, common_footers: set) -> str:
    """
    Removes detected common headers, common footers, and standalone page number artifacts
    from an individual page's text while preserving substantive academic content.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Remove line if it exactly matches a detected recurring header (in the top 3 lines)
        if idx <= 2 and line in common_headers:
            continue

        # Remove line if it exactly matches a detected recurring footer (in the bottom 3 lines)
        if idx >= len(lines) - 3 and line in common_footers:
            continue

        # Remove standalone page indicator lines like: "Page 1", "1 / 15", "- 2 -", "Page 3 of 10"
        if re.match(r'^(page\s*\d+(\s*of\s*\d+)?|\d+\s*/\s*\d+|[-—]\s*\d+\s*[-—]|\d+)$', line, re.IGNORECASE):
            # Only remove if it's near top or bottom of the page
            if idx <= 2 or idx >= len(lines) - 3:
                continue

        cleaned_lines.append(raw_line)

    result = "\n".join(cleaned_lines)
    return clean_page_whitespace(result)


def attempt_ocr_on_page(page, pymupdf_doc=None) -> str:
    """
    Prepares and attempts OCR processing for scanned/image pages.
    Uses pytesseract if the local tesseract engine is available.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render high-resolution pixmap from the page (2x scale for OCR clarity)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        ocr_text = pytesseract.image_to_string(img)
        return clean_page_whitespace(ocr_text)
    except Exception:
        # OCR engine binary (tesseract.exe) is not installed on PATH; return empty
        return ""


def process_quiz_pdf(quiz, rerun: bool = False) -> dict:
    """
    Core Pipeline:
    PDF -> Page extraction -> Text extraction -> Page number preservation -> Clean text -> OCR Preparation
    Stores each page's extracted content in QuizExtractedPage model.
    """
    if not quiz.pdf_file:
        return {
            'success': False,
            'error': 'No PDF file attached to this quiz.'
        }

    pdf_path = quiz.pdf_file.path
    if not os.path.exists(pdf_path):
        return {
            'success': False,
            'error': f'PDF file not found on disk at {pdf_path}'
        }

    # If already processed and not rerun, return existing stats
    existing_pages = quiz.extracted_pages.all()
    if existing_pages.exists() and not rerun:
        return {
            'success': True,
            'total_pages': existing_pages.count(),
            'clean_chars': sum(p.clean_char_count for p in existing_pages),
            'ocr_needed_count': existing_pages.filter(ocr_needed=True).count(),
            'already_processed': True,
        }

    ext = os.path.splitext(pdf_path)[1].lower()
    if ext in ('.pptx', '.ppt', '.docx', '.doc', '.txt'):
        from .ai_quiz_generator import extract_document_content
        res = extract_document_content(pdf_path)
        sections = res.get('sections', [])
        extracted_pages_data = []

        if sections:
            for idx, sec in enumerate(sections):
                page_text = f"{sec.get('title', '')}\n{sec.get('body', '')}".strip()
                if not page_text:
                    continue
                extracted_pages_data.append(
                    QuizExtractedPage(
                        quiz=quiz,
                        page_number=idx + 1,
                        clean_text=page_text,
                        raw_char_count=len(page_text),
                        clean_char_count=len(page_text),
                        extraction_method=ext.replace('.', ''),
                        ocr_needed=False,
                    )
                )

        if not extracted_pages_data and res.get('text'):
            full_text = res.get('text')
            extracted_pages_data.append(
                QuizExtractedPage(
                    quiz=quiz,
                    page_number=1,
                    clean_text=full_text,
                    raw_char_count=len(full_text),
                    clean_char_count=len(full_text),
                    extraction_method=ext.replace('.', ''),
                    ocr_needed=False,
                )
            )

        QuizExtractedPage.objects.filter(quiz=quiz).delete()
        QuizExtractedPage.objects.bulk_create(extracted_pages_data)

        try:
            from .chunker import generate_semantic_chunks
            generate_semantic_chunks(quiz, rerun=True)
        except Exception:
            pass

        return {
            'success': True,
            'total_pages': len(extracted_pages_data),
            'clean_chars': sum(p.clean_char_count for p in extracted_pages_data),
            'ocr_needed_count': 0,
        }

    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)

        # 1. First Pass: extract raw text page-by-page while preserving page index
        raw_page_texts = []
        for page_idx in range(total_pages):
            page = doc[page_idx]
            text = page.get_text("text") or ""
            raw_page_texts.append(text)

        # 2. Detect repeated global headers & footers
        common_headers, common_footers = detect_repeated_headers_footers(raw_page_texts)

        # 3. Second Pass: clean text, perform OCR check if empty, and persist
        extracted_pages_data = []
        ocr_count = 0
        total_clean_chars = 0

        for page_idx in range(total_pages):
            page_number = page_idx + 1  # 1-based page number
            page = doc[page_idx]
            raw_text = raw_page_texts[page_idx]

            clean_text = clean_page_lines(raw_text, common_headers, common_footers)
            extraction_method = 'fitz'
            ocr_needed = False

            # Check if page is empty or scanned (fewer than 30 readable chars)
            if len(clean_text) < 30:
                ocr_text = attempt_ocr_on_page(page, doc)
                if ocr_text and len(ocr_text) >= 30:
                    clean_text = ocr_text
                    extraction_method = 'ocr'
                else:
                    # Page has no extractable text; flag as OCR needed/scanned
                    extraction_method = 'empty_scanned'
                    ocr_needed = True
                    ocr_count += 1

            clean_char_count = len(clean_text)
            total_clean_chars += clean_char_count

            extracted_pages_data.append(
                QuizExtractedPage(
                    quiz=quiz,
                    page_number=page_number,
                    clean_text=clean_text,
                    raw_char_count=len(raw_text),
                    clean_char_count=clean_char_count,
                    extraction_method=extraction_method,
                    ocr_needed=ocr_needed,
                )
            )

        # 4. Save to Database (atomic bulk create or update)
        QuizExtractedPage.objects.filter(quiz=quiz).delete()
        QuizExtractedPage.objects.bulk_create(extracted_pages_data)

        # 5. Phase 3: Trigger Semantic Chunking
        try:
            from .chunker import generate_semantic_chunks
            generate_semantic_chunks(quiz, rerun=True)
        except Exception:
            pass

        return {
            'success': True,
            'total_pages': total_pages,
            'clean_chars': total_clean_chars,
            'ocr_needed_count': ocr_count,
            'pages_created': len(extracted_pages_data),
            'already_processed': False,
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        if doc is not None:
            doc.close()
