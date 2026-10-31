import re
from accounts.models import QuizSemanticChunk

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def count_tokens(text: str) -> int:
    """
    Counts tokens accurately using tiktoken cl100k_base, with fallback
    to whitespace word estimation (~1.33 tokens per English word).
    """
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, int(len(text.split()) * 1.33))


def slice_tokens(text: str, num_tokens: int, from_end: bool = True) -> str:
    """
    Extracts approximately `num_tokens` from either the end or beginning of a text block.
    Used to compute seamless contextual overlap across split semantic chunks.
    """
    if not text or num_tokens <= 0:
        return ""

    if _ENC is not None:
        tokens = _ENC.encode(text, disallowed_special=())
        if from_end:
            sub = tokens[-num_tokens:]
        else:
            sub = tokens[:num_tokens]
        return _ENC.decode(sub)
    else:
        # Word-based fallback approximation (~0.75 words per token)
        words = text.split()
        word_count = max(1, int(num_tokens * 0.75))
        if from_end:
            return " ".join(words[-word_count:])
        return " ".join(words[:word_count])


# Regex patterns identifying structural boundaries:
# 1. Primary Chapters / Units / Modules
RE_CHAPTER = re.compile(
    r'^(chapter|unit|module|part)\s*([0-9ivx]+)?[:.\s\-]+(.+)$',
    re.IGNORECASE
)
# 2. Numbered Section (e.g. "1.1 Introduction", "2.3.1 Relational Calculus")
RE_NUMBERED_SEC = re.compile(
    r'^(\d+\.\d+(\.\d+)?)\s+([A-Z0-9].+)$'
)
# 3. Topic / Major Heading (All caps or Title Case with no trailing punctuation)
RE_TOPIC_HEADING = re.compile(
    r'^[A-Z][A-Za-z0-9\s,\-&()]{3,80}$'
)


def is_heading_candidate(line: str) -> tuple:
    """
    Evaluates whether a single line constitutes a structural heading:
    Returns (is_heading: bool, level: int, title: str)
    level 1 = Chapter / Unit
    level 2 = Numbered Section
    level 3 = Subtopic / Section
    """
    clean = line.strip()
    if not clean or len(clean) > 100:
        return False, 0, ""

    # Check Chapter/Unit
    m = RE_CHAPTER.match(clean)
    if m:
        return True, 1, clean

    # Check Numbered Section (e.g. 1.2 Functional Dependency)
    m = RE_NUMBERED_SEC.match(clean)
    if m:
        return True, 2, clean

    # Check short, title-like lines without ending period/colon
    if (
        clean.isupper()
        and 4 <= len(clean) <= 60
        and not clean.endswith(('.', ':', ';', ','))
    ):
        return True, 3, clean.title()

    if (
        RE_TOPIC_HEADING.match(clean)
        and len(clean.split()) <= 8
        and not clean.endswith(('.', ':', ';', ','))
        and not clean.startswith(('Note:', 'Example:', 'Where', 'Step', 'Table', 'Figure'))
    ):
        return True, 3, clean

    return False, 0, ""


def parse_structural_units(extracted_pages: list, default_topic: str) -> list:
    """
    Parses clean page text into an ordered list of structural units:
    [
      {
        'topic': 'Chapter 1: Process Management',
        'paragraphs': [ 'text block 1...', 'text block 2...' ],
        'pages': {1, 2},
      },
      ...
    ]
    """
    units = []
    current_topic = default_topic
    current_paragraphs = []
    current_pages = set()

    for p in extracted_pages:
        page_num = p.page_number
        text = p.clean_text.strip()
        if not text:
            continue

        # Split into logical paragraphs by double line breaks
        paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]

        for para in paragraphs:
            lines = [l.strip() for l in para.splitlines() if l.strip()]
            if not lines:
                continue

            first_line = lines[0]
            is_head, level, head_text = is_heading_candidate(first_line)

            # If the paragraph starts with a heading and we already have accumulated content:
            if is_head:
                if current_paragraphs:
                    units.append({
                        'topic': current_topic,
                        'paragraphs': current_paragraphs,
                        'pages': set(current_pages),
                    })
                    current_paragraphs = []
                    current_pages = set()

                current_topic = head_text
                current_pages.add(page_num)

                # Remaining lines in this paragraph belong to the new topic
                remaining_para = "\n".join(lines[1:]).strip()
                if remaining_para:
                    current_paragraphs.append(remaining_para)
            else:
                current_paragraphs.append(para)
                current_pages.add(page_num)

    # Append any trailing content
    if current_paragraphs:
        units.append({
            'topic': current_topic,
            'paragraphs': current_paragraphs,
            'pages': set(current_pages),
        })

    return units


def split_oversized_paragraph(para: str, max_tokens: int = 350) -> list:
    """Splits an oversized paragraph into sentence clusters within max_tokens."""
    if count_tokens(para) <= max_tokens:
        return [para]
    raw_sentences = re.split(r'(?<=[.?!])\s+', para)
    sub_blocks = []
    current_block = []
    current_tokens = 0
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current_block:
            sub_blocks.append(" ".join(current_block))
            current_block = [sent]
            current_tokens = sent_tokens
        else:
            current_block.append(sent)
            current_tokens += sent_tokens
    if current_block:
        sub_blocks.append(" ".join(current_block))
    return sub_blocks if sub_blocks else [para]


def assemble_semantic_chunks(structural_units: list, target_min_tokens: int = 800, target_max_tokens: int = 1200, overlap_tokens: int = 150) -> list:
    """
    Assembles structural units into semantic chunks respecting the 800-1200 token budget
    and applying 100-200 token overlap when necessary across topic splits.
    """
    chunks = []
    previous_chunk_tail = ""

    for unit in structural_units:
        topic = unit['topic']
        paragraphs = unit['paragraphs']
        pages = sorted(list(unit['pages']))
        start_page = pages[0] if pages else 1
        end_page = pages[-1] if pages else start_page

        # Combine paragraphs for this topic
        full_unit_text = "\n\n".join(paragraphs).strip()
        unit_tokens = count_tokens(full_unit_text)

        # CASE 1: Fits comfortably within budget (up to target_max_tokens)
        if unit_tokens <= target_max_tokens:
            # If tiny (< 400 tokens) and can merge with previous chunk under same general section, merge
            if (
                chunks
                and (chunks[-1]['token_count'] + unit_tokens) <= target_max_tokens
                and chunks[-1]['end_page'] >= start_page - 1
            ):
                last_chunk = chunks[-1]
                last_chunk['content'] += f"\n\n### {topic}\n{full_unit_text}"
                last_chunk['token_count'] = count_tokens(last_chunk['content'])
                last_chunk['char_count'] = len(last_chunk['content'])
                last_chunk['end_page'] = max(last_chunk['end_page'], end_page)
                # Keep combined title if meaningful
                if topic not in last_chunk['topic']:
                    last_chunk['topic'] = f"{last_chunk['topic']} & {topic}"[:190]
                continue

            # Standalone semantic chunk
            content = full_unit_text
            overlap_applied = 0
            # If continuing from an oversized topic slice, carry contextual overlap
            if previous_chunk_tail:
                overlap_text = f"[Context from prior section]: ...{previous_chunk_tail}...\n\n"
                content = overlap_text + content
                overlap_applied = count_tokens(previous_chunk_tail)
                previous_chunk_tail = ""

            token_cnt = count_tokens(content)
            chunks.append({
                'topic': topic,
                'content': content,
                'token_count': token_cnt,
                'char_count': len(content),
                'start_page': start_page,
                'end_page': end_page,
                'overlap_token_count': overlap_applied,
            })

        # CASE 2: Oversized section (> target_max_tokens) - split at paragraph boundaries with overlap
        else:
            refined_paras = []
            for p in paragraphs:
                if count_tokens(p) > (target_max_tokens // 2):
                    refined_paras.extend(split_oversized_paragraph(p, max_tokens=target_max_tokens // 3))
                else:
                    refined_paras.append(p)

            current_slice_paras = []
            current_slice_tokens = 0

            # Carry previous tail if any
            slice_prefix = ""
            if previous_chunk_tail:
                slice_prefix = f"[Context from prior section]: ...{previous_chunk_tail}...\n\n"
                current_slice_tokens += count_tokens(slice_prefix)

            for para in refined_paras:
                para_tokens = count_tokens(para)

                if current_slice_tokens + para_tokens > target_max_tokens and current_slice_paras:
                    # Finalize current slice
                    slice_content = slice_prefix + "\n\n".join(current_slice_paras)
                    token_cnt = count_tokens(slice_content)
                    chunks.append({
                        'topic': topic,
                        'content': slice_content,
                        'token_count': token_cnt,
                        'char_count': len(slice_content),
                        'start_page': start_page,
                        'end_page': end_page,
                        'overlap_token_count': count_tokens(slice_prefix) if slice_prefix else 0,
                    })

                    # Calculate overlap for next slice from end of current slice
                    overlap_snippet = slice_tokens("\n\n".join(current_slice_paras), overlap_tokens, from_end=True)
                    slice_prefix = f"[Context from previous part]: ...{overlap_snippet}...\n\n"
                    current_slice_paras = [para]
                    current_slice_tokens = count_tokens(slice_prefix) + para_tokens
                else:
                    current_slice_paras.append(para)
                    current_slice_tokens += para_tokens

            # Any remaining paragraphs in oversized unit
            if current_slice_paras:
                slice_content = slice_prefix + "\n\n".join(current_slice_paras)
                token_cnt = count_tokens(slice_content)
                chunks.append({
                    'topic': topic,
                    'content': slice_content,
                    'token_count': token_cnt,
                    'char_count': len(slice_content),
                    'start_page': start_page,
                    'end_page': end_page,
                    'overlap_token_count': count_tokens(slice_prefix) if slice_prefix else 0,
                })
                # Prepare tail for next unit
                previous_chunk_tail = slice_tokens("\n\n".join(current_slice_paras), overlap_tokens, from_end=True)

    return chunks


def generate_semantic_chunks(quiz, rerun: bool = False) -> dict:
    """
    Core Entrypoint for Phase 3:
    Extracted Pages -> Hierarchy Parsing -> Paragraph Grouping -> 800-1200 Token Chunks -> Provenance Persistence
    """
    pages = quiz.extracted_pages.filter(clean_char_count__gt=0).order_by('page_number')
    if not pages.exists():
        return {
            'success': False,
            'error': 'No clean extracted pages found. Run Phase 2 PDF processing first.'
        }

    existing_chunks = quiz.chunks.all()
    if existing_chunks.exists() and not rerun:
        return {
            'success': True,
            'total_chunks': existing_chunks.count(),
            'avg_tokens': int(sum(c.token_count for c in existing_chunks) / existing_chunks.count()),
            'already_chunked': True,
        }

    try:
        # Default topic fallback from quiz title and subject
        default_topic = f"{quiz.subject.name} — Overview"
        if quiz.title:
            default_topic = quiz.title

        # Step 1: Parse structural units (Chapter -> Section -> Subtopic)
        structural_units = parse_structural_units(list(pages), default_topic)

        # Step 2: Assemble into chunks (800-1200 tokens with 100-200 token overlap)
        chunk_dicts = assemble_semantic_chunks(
            structural_units,
            target_min_tokens=800,
            target_max_tokens=1200,
            overlap_tokens=150
        )

        if not chunk_dicts:
            # Fallback if document text was exceptionally brief
            combined_text = "\n\n".join(p.clean_text for p in pages)
            p_list = [p.page_number for p in pages]
            chunk_dicts = [{
                'topic': default_topic,
                'content': combined_text,
                'token_count': count_tokens(combined_text),
                'char_count': len(combined_text),
                'start_page': min(p_list) if p_list else 1,
                'end_page': max(p_list) if p_list else 1,
                'overlap_token_count': 0,
            }]

        # Step 3: Persist to Database with formatted CHK-01 IDs
        db_records = []
        for idx, cd in enumerate(chunk_dicts, start=1):
            chunk_id_str = f"CHK-{idx:02d}"
            db_records.append(
                QuizSemanticChunk(
                    quiz=quiz,
                    chunk_index=idx,
                    chunk_id=chunk_id_str,
                    topic=cd['topic'][:200],
                    content=cd['content'],
                    token_count=cd['token_count'],
                    char_count=cd['char_count'],
                    start_page=cd['start_page'],
                    end_page=cd['end_page'],
                    overlap_token_count=cd['overlap_token_count'],
                )
            )

        QuizSemanticChunk.objects.filter(quiz=quiz).delete()
        QuizSemanticChunk.objects.bulk_create(db_records)

        avg_tok = int(sum(c.token_count for c in db_records) / len(db_records)) if db_records else 0

        return {
            'success': True,
            'total_chunks': len(db_records),
            'avg_tokens': avg_tok,
            'already_chunked': False,
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
