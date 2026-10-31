import json
import re
import os
import requests
from typing import Dict, Any, Tuple, Optional, List
from django.utils import timezone
from .document_extractor.base import NormalizedDocument, EmptyDocumentException
from .text_chunker import SmartChunker, SmartChunk

CORE_SYSTEM_INSTRUCTION = """You are an expert academic educator and study-note generation architect powered by advanced AI.

YOUR MISSION:
Analyze the supplied educational material to identify the core topics, subtopics, concepts, and terminology covered in the document.
For each topic identified in the material, use your full pedagogical intelligence, explanatory depth, and structured reasoning (like Gemini / ChatGPT / Claude) to formulate a COMPREHENSIVE, BEAUTIFULLY STRUCTURED, STUDENT-FRIENDLY MASTER STUDY NOTE.

CRITICAL PRINCIPLES:
1. STRICT TOPIC GROUNDING:
   - Your study note MUST cover ONLY the topics, concepts, and syllabus themes identified in the provided study material.
   - Do NOT introduce unrelated external topics that are not in the document.
   - Ignore administrative metadata (e.g. professor names, college headers, department contacts, syllabus codes).

2. ADVANCED EDUCATIONAL ELABORATION & STRUCTURE:
   - Educational slides and class documents are often terse, fragmented, or bullet-heavy. Do NOT merely copy-paste raw choppy fragments.
   - For every topic identified in the material, provide a complete, clear, student-friendly explanation.
   - Give precise, high-quality definitions for all key terms.
   - Explain the "What", "Why", and "How" of each concept clearly.
   - Break down complex processes, algorithms, or operations into clear numbered steps.
   - Provide classifications, types, or architectures with explanations for each category.
   - Include clear comparisons and contrast tables (e.g., Difference between Item A and Item B) where applicable.
   - Include realistic illustrative examples to help students grasp the ideas.
   - Highlight high-yield exam points, core rules, and formulas.
   - The final result should look like professionally authored, comprehensive lecture notes prepared by a top university professor.
"""

JSON_SCHEMA_INSTRUCTION = """You MUST output ONLY a valid JSON object (no markdown quotes outside JSON, no trailing conversational text) matching this schema:
{
  "title": "Clear Main Topic Title",
  "overview": "1-2 sentence simple introduction of the topic",
  "learning_objectives": ["Objective 1", "Objective 2"],
  "sections": [
    {
      "title": "Subtopic Name",
      "summary": "Short explanation in simple language",
      "key_points": ["Point 1", "Point 2"],
      "definitions": [
        {
          "term": "Term Name",
          "definition": "Short and clear definition",
          "source_reference": "Slide 3 or Page 4"
        }
      ],
      "examples": ["Relevant concise example if present"],
      "steps": ["Step 1: Description", "Step 2: Description"],
      "important_terms": ["Term 1", "Term 2"],
      "comparisons": [
        {
          "item_a": "First Item",
          "item_b": "Second Item",
          "differences": [
            {"feature": "Basis of comparison", "point_a": "...", "point_b": "..."}
          ]
        }
      ],
      "formulas": ["Important formula or syntax if present"],
      "source_references": ["Page/Slide numbers"]
    }
  ],
  "important_definitions": [
    {
      "term": "Term",
      "definition": "Precise, exam-friendly definition"
    }
  ],
  "important_terms": ["Term 1", "Term 2", "Term 3"],
  "exam_points": ["High-yield exam fact 1", "High-yield exam fact 2"],
  "quick_revision": [
    {
      "keyword": "Key Concept",
      "meaning": "One-line revision explanation"
    }
  ],
  "source_references": ["Source pages or slides covered"]
}
Only populate fields that are genuinely present in the source. Omit empty sections."""


def parse_fallback_text_to_structure(raw_text: str, default_title: str = "Study Notes") -> Dict[str, Any]:
    """
    Intelligent fallback parser that transforms any raw text or markdown response
    into the required JSON study-note schema so the user never gets an error.
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    title = default_title
    overview = ""
    sections: List[Dict[str, Any]] = []
    current_sec: Optional[Dict[str, Any]] = None

    for line in lines:
        if line.startswith("# ") and title == default_title:
            title = line[2:].strip()
        elif line.startswith("## "):
            if current_sec:
                sections.append(current_sec)
            current_sec = {
                "title": line[3:].strip(),
                "summary": "",
                "key_points": [],
                "definitions": [],
                "examples": [],
                "steps": [],
                "important_terms": [],
                "comparisons": [],
                "formulas": [],
                "source_references": []
            }
        elif line.startswith("- ") or line.startswith("* "):
            pt = line[2:].strip()
            if current_sec:
                current_sec["key_points"].append(pt)
        elif not overview and not sections and not line.startswith("#"):
            overview = line

    if current_sec:
        sections.append(current_sec)

    return {
        "title": title,
        "overview": overview,
        "learning_objectives": [],
        "sections": sections if sections else [{"title": "Core Revision Points", "key_points": lines[:12]}],
        "important_definitions": [],
        "important_terms": [],
        "exam_points": [],
        "quick_revision": [],
        "source_references": []
    }


def clean_llm_json_response(raw_text: str, default_title: str = "Study Notes") -> Dict[str, Any]:
    """
    Multi-stage, ultra-resilient JSON parser:
    1. Direct json.loads
    2. json_repair.loads (handles unescaped quotes, unclosed objects, bad escapes, truncated JSON)
    3. Regex outer {...} extraction + json_repair
    4. Text-to-structure parser (guarantees a valid structured output without throwing errors)
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 1. Standard json.loads
    try:
        res = json.loads(cleaned)
        if isinstance(res, dict) and res.get("title"):
            return res
    except Exception:
        pass

    # 2. json_repair (handles unescaped quotes, unclosed objects, bad escapes, truncated JSON)
    try:
        import json_repair
        res = json_repair.loads(cleaned)
        if isinstance(res, dict) and res.get("title"):
            return res
    except Exception:
        pass

    # 3. Regex search for outer {...}
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            res = json.loads(candidate)
            if isinstance(res, dict):
                return res
        except Exception:
            pass
        try:
            import json_repair
            res = json_repair.loads(candidate)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    # 4. Fallback text parsing (never throw an unhandled error to the student)
    return parse_fallback_text_to_structure(raw_text, default_title=default_title)


def format_json_to_markdown(data: Dict[str, Any]) -> str:
    """
    Renders structured study note JSON into beautiful, exam-ready Markdown.
    Guarantees robust type checks for all fields.
    """
    if not isinstance(data, dict):
        return str(data)

    md: List[str] = []

    # Title & Overview
    title = data.get("title", "Study Summary")
    md.append(f"# {title}\n")

    overview = data.get("overview", "")
    if isinstance(overview, str) and overview.strip():
        md.append(f"{overview.strip()}\n")

    # Learning Objectives
    objs = data.get("learning_objectives", [])
    if isinstance(objs, list) and objs:
        md.append("## Learning Objectives\n")
        for obj in objs:
            md.append(f"- {obj}")
        md.append("")

    # Sections
    sections = data.get("sections", [])
    if isinstance(sections, list):
        for idx, sec in enumerate(sections, 1):
            if not isinstance(sec, dict):
                if isinstance(sec, str):
                    md.append(f"## {idx}. Section\n- {sec}\n")
                continue

            s_title = sec.get("title", f"Topic {idx}")
            md.append(f"## {idx}. {s_title}\n")

            summary = sec.get("summary", "")
            if isinstance(summary, str) and summary.strip():
                md.append(f"{summary.strip()}\n")

            # Definitions in Section
            defs = sec.get("definitions", [])
            if isinstance(defs, list) and defs:
                md.append("### Definition")
                for d in defs:
                    if isinstance(d, dict):
                        term = d.get("term", "").strip()
                        defn = d.get("definition", "").strip()
                        ref = d.get("source_reference", "")
                        ref_txt = f" *({ref})*" if ref else ""
                        md.append(f"- **{term}:** {defn}{ref_txt}")
                    elif isinstance(d, str):
                        md.append(f"- {d}")
                md.append("")

            # Key Points
            kp = sec.get("key_points", [])
            if isinstance(kp, list) and kp:
                md.append("### Key Points")
                for pt in kp:
                    md.append(f"- {pt}")
                md.append("")

            # Steps / Procedures
            steps = sec.get("steps", [])
            if isinstance(steps, list) and steps:
                md.append("### Process & Steps")
                for s_idx, st in enumerate(steps, 1):
                    clean_st = re.sub(r'^\d+[\.\)]\s*', '', str(st))
                    md.append(f"{s_idx}. {clean_st}")
                md.append("")

            # Formulas / Syntax
            formulas = sec.get("formulas", [])
            if isinstance(formulas, list) and formulas:
                md.append("### Formulas & Syntax")
                for form in formulas:
                    md.append(f"```text\n{form}\n```")
                md.append("")

            # Comparisons
            comparisons = sec.get("comparisons", [])
            if isinstance(comparisons, list):
                for comp in comparisons:
                    if isinstance(comp, dict):
                        item_a = comp.get("item_a", "Item A")
                        item_b = comp.get("item_b", "Item B")
                        diffs = comp.get("differences", [])
                        if isinstance(diffs, list) and diffs:
                            md.append(f"### Difference: {item_a} vs {item_b}\n")
                            md.append(f"| Feature / Basis | {item_a} | {item_b} |")
                            md.append("|---|---|---|")
                            for diff in diffs:
                                if isinstance(diff, dict):
                                    feat = diff.get("feature", "Aspect")
                                    pa = diff.get("point_a", "-")
                                    pb = diff.get("point_b", "-")
                                    md.append(f"| **{feat}** | {pa} | {pb} |")
                            md.append("")

            # Examples
            examples = sec.get("examples", [])
            if isinstance(examples, list) and examples:
                md.append("### Examples")
                for ex in examples:
                    md.append(f"- **Example:** {ex}")
                md.append("")

            # Important terms
            iterms = sec.get("important_terms", [])
            if isinstance(iterms, list) and iterms:
                terms_str = ", ".join([f"`{t}`" for t in iterms if t])
                if terms_str:
                    md.append(f"**Key Terms:** {terms_str}\n")

    # Important Definitions Compilation
    imp_defs = data.get("important_definitions", [])
    if isinstance(imp_defs, list) and imp_defs:
        md.append("## Important Definitions\n")
        for d in imp_defs:
            if isinstance(d, dict):
                t = d.get("term", "").strip()
                df = d.get("definition", "").strip()
                md.append(f"- **{t}:** {df}")
            elif isinstance(d, str):
                md.append(f"- {d}")
        md.append("")

    # High-Yield Exam Points
    exam_pts = data.get("exam_points", [])
    if isinstance(exam_pts, list) and exam_pts:
        md.append("## High-Yield Exam Points\n")
        for ep in exam_pts:
            md.append(f"- {ep}")
        md.append("")

    # Quick Revision Table/List
    quick_rev = data.get("quick_revision", [])
    if isinstance(quick_rev, list) and quick_rev:
        md.append("## Quick Revision\n")
        md.append("| Concept / Keyword | Quick Revision Meaning |")
        md.append("|---|---|")
        for qr in quick_rev:
            if isinstance(qr, dict):
                kw = qr.get("keyword", "")
                mn = qr.get("meaning", "")
                md.append(f"| **{kw}** | {mn} |")
            elif isinstance(qr, str):
                md.append(f"| {qr} | - |")
        md.append("")

    return "\n".join(md).strip()


def synthesize_local_structured_notes(doc: NormalizedDocument) -> Tuple[str, Dict[str, Any], str]:
    """
    Resilient offline/fallback structured educational study note synthesizer.
    Extracts core topics, definitions, bullet points, and exam takeaways directly
    from the document text and formats into the standard NoteForge JSON & Markdown schema.
    """
    text = doc.get_combined_text()
    title = doc.document_title or "Educational Lecture Study Notes"

    sections = []
    if doc.sections:
        for idx, sec in enumerate(doc.sections[:10], 1):
            sec_title = sec.title or f"Concept {idx}"
            sec_text = sec.content.strip()
            if not sec_text:
                continue

            sentences = [s.strip() for s in re.split(r'[.!?]\s+', sec_text) if len(s.strip()) > 15]
            key_pts = [s + "." for s in sentences[:5]]

            defs = []
            for s in sentences:
                for marker in [' is defined as ', ' refers to ', ' is a ', ' is the ']:
                    if marker in s:
                        parts = s.split(marker, 1)
                        term = parts[0].strip()
                        if 2 <= len(term.split()) <= 5:
                            defs.append({
                                "term": term.title(),
                                "definition": (marker.strip() + " " + parts[1].strip() + ".").strip()
                            })
                            break

            sections.append({
                "title": sec_title,
                "summary": sentences[0] + "." if sentences else sec_text[:200],
                "key_points": key_pts,
                "definitions": defs[:3],
                "examples": [],
                "steps": [],
                "important_terms": [d["term"] for d in defs[:4]],
                "formulas": []
            })

    if not sections:
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 20]
        step = max(1, len(sentences) // 4)
        for i in range(0, min(len(sentences), step * 4), step):
            chunk_s = sentences[i:i + step]
            if not chunk_s:
                continue
            sec_name = f"Part {len(sections) + 1}: Key Topics"
            sections.append({
                "title": sec_name,
                "summary": chunk_s[0] + ".",
                "key_points": [s + "." for s in chunk_s[:4]],
                "definitions": [],
                "examples": [],
                "steps": [],
                "important_terms": [],
                "formulas": []
            })

    all_points = []
    for s in sections:
        all_points.extend(s.get("key_points", []))
    exam_points = all_points[:6] if all_points else ["Review core lecture concepts thoroughly."]

    quick_revision = []
    for s in sections:
        for d in s.get("definitions", []):
            quick_revision.append({"keyword": d["term"], "meaning": d["definition"][:120]})
    if not quick_revision:
        for s in sections[:4]:
            quick_revision.append({"keyword": s["title"], "meaning": s["summary"][:100]})

    structured_data = {
        "title": title,
        "overview": f"Comprehensive master study notes covering the educational themes in '{title}'.",
        "learning_objectives": [
            f"Understand the foundational concepts and methodology of {title}.",
            "Master key terminology, architectural steps, and operational characteristics.",
            "Analyze practical examples and high-yield examination takeaways."
        ],
        "sections": sections,
        "important_definitions": [d for s in sections for d in s.get("definitions", [])][:6],
        "important_terms": [t for s in sections for t in s.get("important_terms", [])][:8],
        "exam_points": exam_points,
        "quick_revision": quick_revision,
        "source_references": [doc.source_type.upper() if doc.source_type else "Source Lecture"]
    }

    md_text = format_json_to_markdown(structured_data)
    return md_text, structured_data, "Local Educational Engine (Structured Notes)"


class AISummarizerService:
    """
    Two-Level AI Summarization Service powered by Google Gemini:
    - Level 1: Single pass structured synthesis for standard documents
    - Level 2: Chunk-wise summarization + final synthesis for large multi-unit documents
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        # Prioritize live verified high-speed models with fresh quota
        self.models_to_try = [
            'gemini-flash-lite-latest',
            'gemini-3.5-flash-lite',
            'gemini-3.1-flash-lite',
            'gemini-flash-latest',
        ]

    def _call_gemini(self, prompt: str, temperature: float = 0.2, require_json: bool = True) -> Tuple[Optional[str], str]:
        headers = {'Content-Type': 'application/json'}
        last_error = ""

        for model in self.models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            gen_config: Dict[str, Any] = {
                "temperature": temperature,
                "maxOutputTokens": 8192,
            }
            if require_json:
                gen_config["responseMimeType"] = "application/json"

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": gen_config
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get('candidates', [])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        if parts and 'text' in parts[0]:
                            return parts[0]['text'].strip(), f"Google Gemini ({model})"
                elif resp.status_code == 429:
                    last_error = f"Model {model} rate limited (429)"
                    continue  # Immediately try next model without blocking delay
                else:
                    last_error = f"Model {model} returned HTTP {resp.status_code}"
            except Exception as e:
                last_error = str(e)
                continue

        # If json mode failed across models, try non-json once on top models
        if require_json:
            fallback_text, eng = self._call_gemini(prompt, temperature=temperature, require_json=False)
            if fallback_text:
                return fallback_text, eng

        return None, last_error

    def summarize_document(self, doc: NormalizedDocument) -> Tuple[str, Dict[str, Any], str]:
        """
        Executes Level 1 or Level 2 summarization based on document length and structure.
        Seamlessly falls back to local high-yield educational synthesis if API quotas are exhausted.
        Returns: (markdown_text, structured_json, engine_name)
        """
        combined_text = doc.get_combined_text()
        if not combined_text.strip():
            raise EmptyDocumentException("No readable text found in document.")

        try:
            chunker = SmartChunker(max_chars_per_chunk=14000, single_pass_threshold=22000)
            if chunker.is_single_pass(doc):
                res = self._summarize_single_pass(doc, combined_text)
                if res and res[0].strip():
                    return res
            else:
                res = self._summarize_multi_chunk(doc, chunker)
                if res and res[0].strip():
                    return res
        except Exception:
            pass

        # Resilient Local Educational Synthesis Fallback (Zero 429 error messages shown to user)
        return synthesize_local_structured_notes(doc)

    def _summarize_single_pass(self, doc: NormalizedDocument, text: str) -> Tuple[str, Dict[str, Any], str]:
        prompt = f"""{CORE_SYSTEM_INSTRUCTION}

━━━━━━━━━━━━━━━━━━━━━━
SOURCE MATERIAL METADATA
━━━━━━━━━━━━━━━━━━━━━━
Title: {doc.document_title}
Format: {doc.source_type.upper()}
Total Sections/Units: {doc.total_units}

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━
{JSON_SCHEMA_INSTRUCTION}

━━━━━━━━━━━━━━━━━━━━━━
COMPLETE SOURCE CONTENT
━━━━━━━━━━━━━━━━━━━━━━
{text}
"""
        raw_resp, engine_name = self._call_gemini(prompt, require_json=True)
        structured_data = clean_llm_json_response(raw_resp, default_title=doc.document_title)
        markdown_text = format_json_to_markdown(structured_data)
        if not markdown_text.strip():
            markdown_text = raw_resp
        return markdown_text, structured_data, engine_name

    def _summarize_multi_chunk(self, doc: NormalizedDocument, chunker: SmartChunker) -> Tuple[str, Dict[str, Any], str]:
        chunks = chunker.chunk(doc)
        intermediate_summaries = []
        engine_name_used = "Google Gemini (Multi-Pass)"

        # Step 1: Chunk-wise summarization
        for c in chunks:
            chunk_prompt = f"""{CORE_SYSTEM_INSTRUCTION}

You are analyzing Part {c.chunk_index} of {len(chunks)} ({c.title}) of the document '{doc.document_title}'.
Extract all subtopics, definitions, key points, formulas, processes, and comparisons from this part.
Output valid JSON:
{{
  "chunk_title": "{c.title}",
  "sections": [
    {{
      "title": "Subtopic Name",
      "summary": "Clear summary",
      "key_points": ["Point 1", "Point 2"],
      "definitions": [{{"term": "...", "definition": "..."}}],
      "formulas": [],
      "comparisons": [],
      "exam_points": []
    }}
  ]
}}

CONTENT:
{c.get_text()}
"""
            try:
                raw_chunk, eng = self._call_gemini(chunk_prompt, require_json=True)
                engine_name_used = eng
                c_json = clean_llm_json_response(raw_chunk, default_title=c.title)
                intermediate_summaries.append(c_json)
            except Exception:
                # Include raw chunk text as fallback context
                intermediate_summaries.append({"chunk_title": c.title, "raw_content": c.get_text()[:4000]})

        # Step 2: Final synthesis
        synthesis_prompt = f"""{CORE_SYSTEM_INSTRUCTION}

You are synthesizing the full multi-part analysis of '{doc.document_title}' into the FINAL, COMPLETE, HIGH-YIELD STUDY NOTE.
Do not lose any topics or definitions from the parts. Merge related topics logically and eliminate redundancy.

{JSON_SCHEMA_INSTRUCTION}

EXTRACTED PARTS:
{json.dumps(intermediate_summaries, indent=2)}
"""
        final_raw, eng = self._call_gemini(synthesis_prompt, require_json=True)
        structured_data = clean_llm_json_response(final_raw, default_title=doc.document_title)
        markdown_text = format_json_to_markdown(structured_data)
        if not markdown_text.strip():
            markdown_text = final_raw
        return markdown_text, structured_data, f"{eng} (Multi-Stage Synthesis)"
