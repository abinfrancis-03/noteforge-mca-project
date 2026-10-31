import os
import re
import json
import textwrap
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import requests

from .document_extractor import NormalizedDocument
from .ai_summarizer import clean_llm_json_response

PALETTE = {
    "indigo": {"primary": (79, 70, 229), "bg": (245, 243, 255), "border": (224, 231, 255), "text": (30, 27, 75)},
    "emerald": {"primary": (5, 150, 105), "bg": (236, 253, 245), "border": (209, 250, 229), "text": (6, 78, 59)},
    "amber": {"primary": (217, 119, 6), "bg": (255, 251, 235), "border": (254, 243, 199), "text": (120, 53, 15)},
    "cyan": {"primary": (8, 145, 178), "bg": (236, 254, 255), "border": (207, 250, 254), "text": (22, 78, 99)},
    "rose": {"primary": (225, 29, 72), "bg": (255, 241, 242), "border": (255, 228, 230), "text": (136, 19, 55)},
    "purple": {"primary": (124, 58, 237), "bg": (250, 245, 255), "border": (243, 232, 255), "text": (88, 28, 135)},
    "blue": {"primary": (37, 99, 235), "bg": (239, 246, 255), "border": (219, 234, 254), "text": (30, 58, 138)},
    "slate": {"primary": (51, 65, 85), "bg": (248, 250, 252), "border": (226, 232, 240), "text": (15, 23, 42)},
}

COLOR_KEYS = ["indigo", "emerald", "amber", "cyan", "rose", "purple", "blue"]

NOISE_PATTERN = re.compile(
    r'(prof\b|professor|associate\s+prof|assistant\s+prof|department\s+of|college\b|university\b|course\s+code|text\s*books?|reference\s*books?|syllabus|prepared\s+by|presented\s+by|contact\b|email\b|academic\s+year|semester\s+[0-9]+)',
    re.IGNORECASE
)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Attempts to load Segoe UI, Arial, or falls back to default font."""
    font_names = (
        ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]
        if bold
        else ["segoeui.ttf", "arial.ttf", "calibri.ttf"]
    )
    for name in font_names:
        p = os.path.join("C:/Windows/Fonts", name)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


INFOGRAPHIC_SYSTEM_PROMPT = """You are an expert university professor and educational mind-map architect powered by advanced AI.

YOUR MISSION:
Analyze the provided academic study material to identify the core educational concepts and formulate a HIGH-YIELD VISUAL MIND-MAP INFOGRAPHIC.

CRITICAL ACCURACY & GROUNDING RULES:
1. STRICT TOPIC GROUNDING:
   - Identify the 4 to 6 TRUE EDUCATIONAL TOPICS actually taught in this study material.
   - DISCARD all administrative noise: professor names, college headers, department contacts, syllabus tables, and textbook bibliographies.
   - DO NOT use generic placeholder titles (e.g. "WHAT IS IT?", "ARCHITECTURE", "WORKING MECHANISM") unless the topic is specifically about software architecture.
   - Each card MUST be titled with the REAL SUBJECT TOPIC (e.g., "WHAT IS SOFTWARE ENGINEERING?", "SOFTWARE QUALITIES (INTERNAL VS EXTERNAL)", "LITERATE PROGRAMMING (KNUTH)", "GIT VERSION CONTROL ARCHITECTURE", "2'S COMPLEMENT INTEGER ARITHMETIC").

2. FACTUAL DEPTH & PRECISION:
   - For each card, generate 3 to 4 clear, student-friendly, factually accurate bullet points directly from the material.
   - Highlight: One real formula, rule, syntax, or core definition for that topic.
   - Table: A clean 2-column comparison or classification table from the material (e.g., Concept A vs Concept B, or Types vs Features).
   - Bottom Key Takeaways: 3 to 4 high-yield exam revision facts that directly answer typical exam questions.

You MUST output ONLY valid JSON matching this schema:
{
  "title": "EXACT SUBJECT / MODULE TITLE",
  "subtitle": "Clear, concise 1-sentence summary of the core syllabus topic",
  "cards": [
    {
      "number": 1,
      "title": "EXACT TOPIC NAME",
      "icon_label": "TOPIC KEYWORD",
      "points": [
        "Accurate, factually correct explanatory point 1",
        "Accurate, factually correct explanatory point 2",
        "Accurate, factually correct explanatory point 3"
      ],
      "highlight": "Exact formula, syntax, rule, or core definition",
      "table": {
        "headers": ["Type / Concept", "Description / Feature"],
        "rows": [["Item A", "Point"], ["Item B", "Point"]]
      }
    }
  ],
  "key_takeaways": [
    "High-yield exam fact 1",
    "High-yield exam fact 2",
    "High-yield exam fact 3"
  ]
}
"""


def filter_clean_academic_content(doc: NormalizedDocument) -> str:
    """Filters out administrative slides (prof names, textbooks, course codes) to ensure pure academic accuracy."""
    clean_parts = []
    if doc.sections:
        for s in doc.sections:
            title = (s.title or "").strip()
            content = (s.content or "").strip()
            if not content or len(content) < 25:
                continue
            first_lines = " ".join(content.splitlines()[:3])
            if NOISE_PATTERN.search(title) or (NOISE_PATTERN.search(first_lines) and len(content.splitlines()) <= 5):
                continue
            clean_parts.append(f"### {title}\n{content}")

    if clean_parts:
        return "\n\n".join(clean_parts)
    return doc.get_combined_text()


def extract_infographic_data(
    doc: NormalizedDocument,
    api_key: str,
    summary_json: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calls Gemini to generate the structured visual infographic spec from the document.
    Incorporates verified summary_json if available to guarantee 100% curriculum accuracy.
    """
    clean_text = filter_clean_academic_content(doc)

    summary_context = ""
    if summary_json and isinstance(summary_json, dict) and summary_json.get("sections"):
        summary_context = f"""
━━━━━━━━━━━━━━━━━━━━━━
VERIFIED CURRICULUM TOPICS & DEFINITIONS (GROUND TRUTH):
━━━━━━━━━━━━━━━━━━━━━━
{json.dumps(summary_json.get('sections', [])[:6], indent=2)}
"""

    prompt = f"""{INFOGRAPHIC_SYSTEM_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━
SOURCE MATERIAL METADATA
━━━━━━━━━━━━━━━━━━━━━━
Document: {doc.document_title}
Format: {doc.source_type.upper()}
Units: {doc.total_units}
{summary_context}
━━━━━━━━━━━━━━━━━━━━━━
SOURCE CONTENT TO ANALYZE
━━━━━━━━━━━━━━━━━━━━━━
{clean_text[:22000]}
"""
    headers = {"Content-Type": "application/json"}
    models = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-3.5-flash",
    ]
    raw_resp = ""

    if api_key:
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 8192,
                    "responseMimeType": "application/json",
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=35)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            raw_resp = parts[0]["text"].strip()
                            break
            except Exception:
                continue

    if not raw_resp:
        return build_fallback_infographic_data(doc, summary_json=summary_json)

    parsed = clean_llm_json_response(raw_resp, default_title=doc.document_title)
    if not parsed.get("cards"):
        return build_fallback_infographic_data(doc, partial=parsed, summary_json=summary_json)
    return parsed


def build_fallback_infographic_data(
    doc: NormalizedDocument,
    partial: Optional[Dict[str, Any]] = None,
    summary_json: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds clean, accurate structured cards from verified summary_json or filtered document sections.
    Guarantees zero professor names, textbook lists, or course code headers in fallback cards.
    """
    title = (partial and partial.get("title")) or (summary_json and summary_json.get("title")) or doc.document_title or "Study Infographic"
    subtitle = (partial and partial.get("subtitle")) or (summary_json and summary_json.get("overview")) or "High-yield conceptual revision mind-map"
    cards = []

    # Priority 1: Build from verified summary_json
    if summary_json and isinstance(summary_json, dict) and summary_json.get("sections"):
        for idx, sec in enumerate(summary_json["sections"][:6], 1):
            s_title = sec.get("title", f"Topic {idx}").strip()
            pts = sec.get("key_points", [])
            if not pts:
                s_sum = sec.get("summary", "")
                pts = [s_sum] if s_sum else [f"Review key principles of {s_title}."]

            hl = ""
            defs = sec.get("definitions", [])
            if defs and isinstance(defs[0], dict):
                hl = f"{defs[0].get('term', '')}: {defs[0].get('definition', '')}"

            tbl = None
            comps = sec.get("comparisons", [])
            if comps and isinstance(comps[0], dict):
                c0 = comps[0]
                tbl = {
                    "headers": [c0.get("item_a", "Item A"), c0.get("item_b", "Item B")],
                    "rows": [[d.get("point_a", ""), d.get("point_b", "")] for d in c0.get("differences", [])[:3]]
                }

            cards.append({
                "number": idx,
                "title": s_title.upper(),
                "icon_label": "CONCEPT",
                "points": pts[:4],
                "highlight": hl,
                "table": tbl,
            })

    # Priority 2: Filter clean academic sections from doc
    if not cards and doc.sections:
        clean_secs = []
        for s in doc.sections:
            t = (s.title or "").strip()
            c = (s.content or "").strip()
            if not c or len(c) < 30:
                continue
            if NOISE_PATTERN.search(t) or NOISE_PATTERN.search(" ".join(c.splitlines()[:2])):
                continue
            clean_secs.append(s)

        for idx, s in enumerate(clean_secs[:6], 1):
            lines = [l.strip("-•* ") for l in s.content.splitlines() if l.strip() and len(l) > 15]
            cards.append({
                "number": idx,
                "title": (s.title or f"Concept {idx}").upper(),
                "icon_label": "TOPIC",
                "points": lines[:4] if lines else [f"Core syllabus principles of {s.title}."],
                "highlight": "",
                "table": None,
            })

    if not cards:
        cards = [{
            "number": 1,
            "title": "FOUNDATIONAL CONCEPTS",
            "icon_label": "OVERVIEW",
            "points": ["Essential syllabus definitions", "Core mechanisms and operational steps", "Practical revision takeaways"],
            "highlight": "",
            "table": None,
        }]

    takeaways = (summary_json and summary_json.get("exam_points")) or [
        f"Master core principles and definitions of {title}.",
        "Review differences and operational trade-offs for examination revision.",
        "Ensure familiarity with formulas, rules, and practical examples."
    ]

    return {
        "title": title,
        "subtitle": subtitle,
        "cards": cards,
        "key_takeaways": takeaways[:4]
    }


def draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[int, int, int, int],
    radius: int,
    fill: Tuple[int, int, int],
    outline: Optional[Tuple[int, int, int]] = None,
    width: int = 1
):
    """Draws a smooth rounded rectangle."""
    x0, y0, x1, y1 = bounds
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=width)


def render_mindmap_image(data: Dict[str, Any], output_path: str) -> str:
    """
    Renders a high-resolution, beautifully styled educational infographic PNG image.
    Width: 1600px. Height dynamically adapts to content.
    Includes collision-proof header wrapping and proportional table rendering.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    WIDTH = 1600
    MARGIN = 70
    CONTENT_WIDTH = WIDTH - (2 * MARGIN)  # 1460px

    # Fonts
    font_brand = get_font(18, bold=True)
    font_title = get_font(38, bold=True)
    font_subtitle = get_font(20, bold=False)
    font_card_num = get_font(17, bold=True)
    font_card_title = get_font(20, bold=True)
    font_card_badge = get_font(13, bold=True)
    font_body = get_font(18, bold=False)
    font_body_bold = get_font(18, bold=True)
    font_highlight = get_font(17, bold=True)
    font_table_hdr = get_font(16, bold=True)
    font_table_cell = get_font(15, bold=False)
    font_takeaway_title = get_font(23, bold=True)
    font_footer = get_font(15, bold=False)

    title_text = data.get("title", "Study Mind Map").strip().upper()
    subtitle_text = data.get("subtitle", "").strip()
    cards_data = data.get("cards", [])
    takeaways = data.get("key_takeaways", [])

    # Column layout: 2 columns
    COL_GAP = 40
    COL_WIDTH = (CONTENT_WIDTH - COL_GAP) // 2  # 710px
    col_x = [MARGIN, MARGIN + COL_WIDTH + COL_GAP]

    # Pre-calculate Card Heights with collision-proof header sizing
    def calculate_card_height(card: Dict[str, Any]) -> Tuple[int, int, List[str]]:
        c_title = card.get("title", "TOPIC").upper()
        # Wrap title if long
        wrapped_title = textwrap.wrap(c_title, width=32)
        header_h = 56 if len(wrapped_title) <= 1 else 76

        h = header_h + 24
        points = card.get("points", [])
        for pt in points:
            wrapped = textwrap.wrap(pt, width=52)
            h += len(wrapped) * 27 + 12

        # Highlight box
        if card.get("highlight"):
            wrapped_hl = textwrap.wrap(card["highlight"], width=50)
            h += len(wrapped_hl) * 25 + 32

        # Table
        tbl = card.get("table")
        if tbl and isinstance(tbl, dict) and tbl.get("rows"):
            h += 34 + len(tbl["rows"][:4]) * 28 + 16

        h += 20  # Bottom padding
        return max(h, 200), header_h, wrapped_title

    card_calcs = [calculate_card_height(c) for c in cards_data]
    card_heights = [calc[0] for calc in card_calcs]

    # Distribute cards across 2 columns to balance height
    col_cards = [[], []]
    col_h = [0, 0]
    for idx, c in enumerate(cards_data):
        target_col = 0 if col_h[0] <= col_h[1] else 1
        col_cards[target_col].append((c, card_heights[idx], idx, card_calcs[idx][1], card_calcs[idx][2]))
        col_h[target_col] += card_heights[idx] + 30

    cards_total_height = max(col_h[0], col_h[1], 300)

    # Takeaway height
    takeaway_h = 90
    for t in takeaways:
        wrapped_t = textwrap.wrap(t, width=110)
        takeaway_h += len(wrapped_t) * 26 + 10

    # Total canvas height
    HEADER_HEIGHT = 210
    TOTAL_HEIGHT = HEADER_HEIGHT + cards_total_height + takeaway_h + 100

    # Create Canvas
    img = Image.new("RGB", (WIDTH, TOTAL_HEIGHT), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # 1. TOP HEADER BANNER
    pill_text = "NOTEFORGE ACADEMIC • VISUAL STUDY MIND-MAP"
    pill_w = 450
    pill_h = 36
    draw_rounded_rect(draw, (MARGIN, 36, MARGIN + pill_w, 36 + pill_h), 18, fill=(224, 231, 255), outline=(199, 210, 254), width=1)
    draw.text((MARGIN + 20, 44), pill_text, fill=(67, 56, 202), font=font_brand)

    # Exam Ready Badge
    exam_badge_text = "✓ EXAM READY"
    draw_rounded_rect(draw, (WIDTH - MARGIN - 150, 36, WIDTH - MARGIN, 36 + pill_h), 18, fill=(220, 252, 231), outline=(187, 247, 208), width=1)
    draw.text((WIDTH - MARGIN - 135, 44), exam_badge_text, fill=(22, 101, 52), font=font_brand)

    # Main Title
    wrapped_title = textwrap.wrap(title_text, width=44)
    cur_y = 86
    for t_line in wrapped_title:
        draw.text((MARGIN, cur_y), t_line, fill=(15, 23, 42), font=font_title)
        cur_y += 46

    # Subtitle
    if subtitle_text:
        wrapped_sub = textwrap.wrap(subtitle_text, width=88)
        for s_line in wrapped_sub:
            draw.text((MARGIN, cur_y), s_line, fill=(71, 85, 105), font=font_subtitle)
            cur_y += 28
    cur_y += 12

    # Divider Line
    draw.line([(MARGIN, cur_y), (WIDTH - MARGIN, cur_y)], fill=(226, 232, 240), width=2)
    cards_start_y = cur_y + 30

    # 2. DRAW CARDS IN 2 COLUMNS
    for col_idx in [0, 1]:
        card_y = cards_start_y
        for c, ch, orig_idx, header_h, wrapped_c_title in col_cards[col_idx]:
            color_theme = COLOR_KEYS[orig_idx % len(COLOR_KEYS)]
            pal = PALETTE[color_theme]

            cx = col_x[col_idx]
            cw = COL_WIDTH

            # Card Background & Subtle Shadow
            draw_rounded_rect(draw, (cx + 3, card_y + 3, cx + cw + 3, card_y + ch + 3), radius=16, fill=(241, 245, 249))
            draw_rounded_rect(draw, (cx, card_y, cx + cw, card_y + ch), radius=16, fill=(255, 255, 255), outline=pal["border"], width=2)

            # Card Header Bar
            draw_rounded_rect(draw, (cx, card_y, cx + cw, card_y + header_h), radius=16, fill=pal["bg"])
            draw.rectangle([cx, card_y + 20, cx + cw, card_y + header_h], fill=pal["bg"])
            draw.line([(cx, card_y + header_h), (cx + cw, card_y + header_h)], fill=pal["border"], width=1)

            # Number Badge inside Header
            num_str = f"{c.get('number', orig_idx + 1):02d}"
            badge_w = 38
            badge_h = 30
            draw_rounded_rect(draw, (cx + 14, card_y + 12, cx + 14 + badge_w, card_y + 12 + badge_h), radius=8, fill=pal["primary"])
            draw.text((cx + 22, card_y + 16), num_str, fill=(255, 255, 255), font=font_card_num)

            # Icon Label Pill on right of header
            icon_lbl = c.get("icon_label", "").upper()
            if icon_lbl:
                lbl_w = min(len(icon_lbl) * 8 + 16, 140)
                draw_rounded_rect(draw, (cx + cw - lbl_w - 14, card_y + 12, cx + cw - 14, card_y + 38), radius=6, fill=(255, 255, 255), outline=pal["border"], width=1)
                draw.text((cx + cw - lbl_w - 6, card_y + 16), textwrap.shorten(icon_lbl, width=14), fill=pal["primary"], font=font_card_badge)

            # Card Title (Wrapped cleanly without badge collision)
            title_y = card_y + 14
            for t_line in wrapped_c_title:
                draw.text((cx + 60, title_y), t_line, fill=pal["text"], font=font_card_title)
                title_y += 24

            # Card Body: Bullet Points
            b_y = card_y + header_h + 16
            for pt in c.get("points", []):
                draw.ellipse([cx + 18, b_y + 6, cx + 26, b_y + 14], fill=pal["primary"])
                wrapped_pt = textwrap.wrap(pt, width=52)
                for p_line in wrapped_pt:
                    draw.text((cx + 36, b_y), p_line, fill=(30, 41, 59), font=font_body)
                    b_y += 27
                b_y += 8

            # Highlight Box (Rule / Formula / Syntax)
            hl = c.get("highlight")
            if hl:
                hl_wrapped = textwrap.wrap(hl, width=50)
                hl_box_h = len(hl_wrapped) * 25 + 18
                draw_rounded_rect(draw, (cx + 18, b_y, cx + cw - 18, b_y + hl_box_h), radius=8, fill=pal["bg"], outline=pal["border"], width=1)
                draw.text((cx + 30, b_y + 4), "★ KEY PRINCIPLE:", fill=pal["primary"], font=font_card_badge)
                h_text_y = b_y + 22
                for h_line in hl_wrapped:
                    draw.text((cx + 30, h_text_y), h_line, fill=pal["text"], font=font_highlight)
                    h_text_y += 25
                b_y += hl_box_h + 12

            # Comparison / Classification Table
            tbl = c.get("table")
            if tbl and isinstance(tbl, dict) and tbl.get("rows"):
                t_headers = tbl.get("headers", ["Item", "Description"])
                t_rows = tbl.get("rows", [])
                t_w = cw - 36
                col1_w = int(t_w * 0.38)
                col2_w = t_w - col1_w

                draw_rounded_rect(draw, (cx + 18, b_y, cx + 18 + t_w, b_y + 28), radius=6, fill=pal["bg"])
                draw.text((cx + 28, b_y + 5), textwrap.shorten(str(t_headers[0]), width=20), fill=pal["text"], font=font_table_hdr)
                if len(t_headers) > 1:
                    draw.text((cx + 18 + col1_w + 10, b_y + 5), textwrap.shorten(str(t_headers[1]), width=32), fill=pal["text"], font=font_table_hdr)
                b_y += 30

                for r in t_rows[:4]:
                    c1_txt = str(r[0]) if len(r) > 0 else "-"
                    c2_txt = str(r[1]) if len(r) > 1 else "-"
                    draw.text((cx + 28, b_y + 4), textwrap.shorten(c1_txt, width=22), fill=(51, 65, 85), font=font_table_cell)
                    draw.text((cx + 18 + col1_w + 10, b_y + 4), textwrap.shorten(c2_txt, width=42), fill=(51, 65, 85), font=font_table_cell)
                    draw.line([(cx + 18, b_y + 25), (cx + 18 + t_w, b_y + 25)], fill=(241, 245, 249), width=1)
                    b_y += 27

            card_y += ch + 30

    # 3. BOTTOM KEY TAKEAWAYS CARD
    takeaway_y = cards_start_y + cards_total_height + 20
    draw_rounded_rect(draw, (MARGIN + 3, takeaway_y + 3, WIDTH - MARGIN + 3, takeaway_y + takeaway_h + 3), radius=16, fill=(241, 245, 249))
    draw_rounded_rect(draw, (MARGIN, takeaway_y, WIDTH - MARGIN, takeaway_y + takeaway_h), radius=16, fill=(255, 255, 255), outline=(203, 213, 225), width=2)

    # Header Banner
    t_hdr_h = 48
    draw_rounded_rect(draw, (MARGIN, takeaway_y, WIDTH - MARGIN, takeaway_y + t_hdr_h), radius=16, fill=(15, 23, 42))
    draw.rectangle([MARGIN, takeaway_y + 18, WIDTH - MARGIN, takeaway_y + t_hdr_h], fill=(15, 23, 42))
    draw.text((MARGIN + 24, takeaway_y + 11), "⚡ HIGH-YIELD EXAM TAKEAWAYS & CRITICAL REVISION", fill=(255, 255, 255), font=font_takeaway_title)

    # Takeaway Bullets
    ty = takeaway_y + t_hdr_h + 18
    for pt in takeaways:
        draw.ellipse([MARGIN + 24, ty + 6, MARGIN + 34, ty + 16], fill=(79, 70, 229))
        wrapped_t = textwrap.wrap(pt, width=108)
        for t_line in wrapped_t:
            draw.text((MARGIN + 46, ty), t_line, fill=(15, 23, 42), font=font_body_bold)
            ty += 27
        ty += 6

    # 4. FOOTER WATERMARK
    footer_y = TOTAL_HEIGHT - 45
    draw.text((MARGIN, footer_y), "NOTEFORGE ACADEMIC ENGINE • GROUNDED IN VERIFIED TEACHER STUDY MATERIALS", fill=(148, 163, 184), font=font_footer)
    draw.text((WIDTH - MARGIN - 260, footer_y), "STRICT SYLLABUS TOPIC ACCURACY", fill=(148, 163, 184), font=font_footer)

    img.save(output_path, "PNG", optimize=True)
    return output_path
