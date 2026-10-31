"""
Important Concept Ranking Engine (Phase 5 — Noteforge AI MCQ System)

Technique:
Educational content prioritization

Methods:
Assign an importance score based on:
- Core concept
- Academic relevance
- Relationships
- Applications
- Repetition/emphasis
- Exam usefulness

Tiers:
- HIGH: Core curricular concepts, formal definitions, axioms, exam-critical topics
- MEDIUM: Supporting mechanisms, intermediate applications, comparative distinctions
- LOW: Background history, superficial trivia, peripheral narrative context

Prompt:
From the extracted concepts, rank the concepts according to their educational importance.
Prioritize core concepts, important definitions, relationships, processes, applications, comparisons
and concepts useful for examination preparation. Do not prioritize trivial facts. Return each concept
with an importance level and reason based only on the source material.
"""

import os
import re
import json
import logging
from django.conf import settings
from accounts.models import Quiz, QuizRankedConcept
from .ai_analyzer import run_quiz_ai_analysis, get_api_key

logger = logging.getLogger(__name__)

RANKING_SYSTEM_PROMPT = """You are a senior academic curriculum evaluator and university exam question setter for Noteforge.
Your job is to prioritize educational concepts so that generated assessment questions are genuinely valuable for student learning and exam preparation.

EVALUATION CRITERIA:
1. Core Concept: Is this a foundational pillar of the subject matter?
2. Academic Relevance: Is it required curriculum knowledge?
3. Relationships: Does it link, govern, or constrain multiple entities?
4. Applications: Does it have direct practical problem-solving use?
5. Repetition & Emphasis: Is it central throughout the source document?
6. Exam Usefulness: Would an instructor prioritize this on an examination over superficial trivia?

IMPORTANCE TIERS:
- HIGH (score 75-100): Critical principles, formal axioms, primary definitions, key processes. Guaranteed exam relevance.
- MEDIUM (score 50-74): Useful supporting concepts, secondary comparisons, practical implementations.
- LOW (score 1-49): Background history, introductory fluff, peripheral trivia not central to exam mastery.

STRICT INSTRUCTIONS:
- Base reasoning ONLY on the provided source material.
- Do NOT hallucinate outside facts.
- Assign every concept: "concept", "category", "importance" ("HIGH"|"MEDIUM"|"LOW"), "importance_score" (int 1-100), "reason", "chunk_id", and "pages".
- Return ONLY a JSON array.
"""

RANKING_USER_PROMPT = """From the extracted concepts, rank the concepts according to their educational importance. Prioritize core concepts, important definitions, relationships, processes, applications, comparisons and concepts useful for examination preparation. Do not prioritize trivial facts. Return each concept with an importance level and reason based only on the source material.

Required Output Schema:
[
  {
    "concept": "string (name of concept)",
    "category": "Core Concept | Definition | Process | Formula | Comparison | Application",
    "importance": "HIGH | MEDIUM | LOW",
    "importance_score": 90,
    "reason": "Clear explanation grounded strictly in the source text of why this concept holds this priority level for exam preparation.",
    "chunk_id": "CHK-01",
    "pages": [10, 11]
  }
]
"""


def extract_with_gemini_ranking(concepts_payload: str, api_key: str) -> tuple[list, str]:
    """Uses Google Gemini to rank educational concepts."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model_candidates = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-pro']
    last_err = None

    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=RANKING_SYSTEM_PROMPT,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                }
            )

            prompt = f"{RANKING_USER_PROMPT}\n\nEXTRACTED CONCEPTS PAYLOAD:\n{concepts_payload}"
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())
            if isinstance(parsed, dict) and 'ranked_concepts' in parsed:
                parsed = parsed['ranked_concepts']
            if isinstance(parsed, dict) and 'concepts' in parsed:
                parsed = parsed['concepts']

            return parsed, model_name

        except Exception as e:
            logger.warning(f"Concept ranking failed with model {model_name}: {e}")
            last_err = e
            continue

    raise RuntimeError(f"All Gemini models failed ranking. Last error: {last_err}")


def fallback_deterministic_ranking(analysis_data: dict, quiz: Quiz) -> list:
    """
    Deterministic educational prioritization engine.
    Applies scoring heuristics based on Core Concept, Academic Relevance,
    Relationships, Applications, and Exam Usefulness.
    """
    ranked = []
    seen_concepts = set()

    # High exam yield keywords
    exam_critical_kw = [
        'normal form', 'normalization', 'dependency', 'functional dependency',
        'transitive', 'candidate key', 'primary key', 'axiom', 'armstrong',
        'process', 'thread', 'scheduling', 'deadlock', 'semaphore', 'mutex',
        'bcnf', '1nf', '2nf', '3nf', '4nf', '5nf', 'acid', 'transaction',
        'anomaly', 'lossless', 'decomposition'
    ]

    # Trivia / Low importance keywords
    low_priority_kw = [
        'history', 'invented by', 'named after', 'background', 'overview',
        'originated', 'timeline', 'etymology', 'preliminary', 'introduction',
        'briefly', 'historical'
    ]

    # 1. Evaluate Formulas and Inference Rules (Inherently High Exam Relevance)
    for f in analysis_data.get('formulas', []):
        name = f.get('name_or_rule', 'Formula / Inference Rule')
        if name.lower() in seen_concepts:
            continue
        seen_concepts.add(name.lower())

        expr = f.get('expression_or_rule', '')
        score = 92
        reason = f"Essential mathematical/inference rule ({expr}). Crucial for analytical and computational questions on examinations."
        ranked.append({
            'concept': name,
            'category': 'Formula / Rule',
            'importance': 'HIGH',
            'importance_score': score,
            'reason': reason,
            'chunk_id': f.get('chunk_id', 'CHK-01'),
            'pages': f.get('pages', [1]),
        })

    # 2. Evaluate Processes and Step-by-Step Procedures
    for p in analysis_data.get('processes', []):
        name = p.get('name', 'Procedural Algorithm')
        if name.lower() in seen_concepts:
            continue
        seen_concepts.add(name.lower())

        steps_count = len(p.get('steps', []))
        score = 88
        reason = f"Step-by-step procedural workflow ({steps_count} steps). High exam probability for methodology and application questions."
        ranked.append({
            'concept': name,
            'category': 'Process',
            'importance': 'HIGH',
            'importance_score': score,
            'reason': reason,
            'chunk_id': p.get('chunk_id', 'CHK-01'),
            'pages': p.get('pages', [1]),
        })

    # 3. Evaluate Definitions
    for d in analysis_data.get('definitions', []):
        term = d.get('term', '')
        if not term or term.lower() in seen_concepts:
            continue
        seen_concepts.add(term.lower())

        t_lower = term.lower()
        is_exam_critical = any(kw in t_lower for kw in exam_critical_kw)
        is_low_priority = any(kw in t_lower for kw in low_priority_kw)

        if is_low_priority:
            score = 35
            tier = 'LOW'
            reason = "Introductory or historical context with minimal problem-solving or testing utility."
        elif is_exam_critical:
            score = 90
            tier = 'HIGH'
            reason = f"Standard foundational terminology tested heavily in academic assessments: {d.get('definition', '')[:120]}..."
        else:
            score = 68
            tier = 'MEDIUM'
            reason = f"Important domain vocabulary required for comprehension of broader units: {d.get('definition', '')[:120]}..."

        ranked.append({
            'concept': term,
            'category': 'Definition',
            'importance': tier,
            'importance_score': score,
            'reason': reason,
            'chunk_id': d.get('chunk_id', 'CHK-01'),
            'pages': d.get('pages', [1]),
        })

    # 4. Evaluate Comparisons
    for c in analysis_data.get('comparisons', []):
        name = f"{c.get('item_a', '')} vs {c.get('item_b', '')}"
        if name.lower() in seen_concepts:
            continue
        seen_concepts.add(name.lower())

        score = 82
        reason = f"Classic comparative exam topic: {c.get('distinction', '')[:140]}..."
        ranked.append({
            'concept': name,
            'category': 'Comparison',
            'importance': 'HIGH',
            'importance_score': score,
            'reason': reason,
            'chunk_id': c.get('chunk_id', 'CHK-01'),
            'pages': c.get('pages', [1]),
        })

    # 5. Evaluate Important Concepts
    for ic in analysis_data.get('important_concepts', []):
        c_title = ic.get('concept', '')
        if not c_title or c_title.lower() in seen_concepts:
            continue
        seen_concepts.add(c_title.lower())

        c_lower = c_title.lower()
        is_exam_critical = any(kw in c_lower for kw in exam_critical_kw)
        is_low_priority = any(kw in c_lower for kw in low_priority_kw)

        if is_low_priority:
            score = 30
            tier = 'LOW'
            reason = "Background material with low likelihood of appearing on examinations."
        elif is_exam_critical:
            score = 94
            tier = 'HIGH'
            reason = f"Core theoretical principle directly addressing primary curriculum goals: {ic.get('significance', 'Fundamental concept.')}"
        else:
            score = 65
            tier = 'MEDIUM'
            reason = f"Supporting concept reinforcing understanding of main topics: {ic.get('explanation', '')[:120]}..."

        ranked.append({
            'concept': c_title,
            'category': 'Core Concept',
            'importance': tier,
            'importance_score': score,
            'reason': reason,
            'chunk_id': ic.get('chunk_id', 'CHK-01'),
            'pages': ic.get('pages', [1]),
        })

    # 6. Evaluate Applications
    for a in analysis_data.get('applications', []):
        area = a.get('area', 'Practical Application')
        if area.lower() in seen_concepts:
            continue
        seen_concepts.add(area.lower())

        score = 70
        reason = f"Real-world and system design application: {a.get('description', '')[:130]}..."
        ranked.append({
            'concept': area,
            'category': 'Application',
            'importance': 'MEDIUM',
            'importance_score': score,
            'reason': reason,
            'chunk_id': a.get('chunk_id', 'CHK-01'),
            'pages': a.get('pages', [1]),
        })

    # Sort descending by score
    ranked.sort(key=lambda x: x['importance_score'], reverse=True)
    return ranked


def rank_quiz_concepts(quiz: Quiz, force: bool = False) -> dict:
    """
    Main entrypoint for Phase 5:
    Reads Phase 4 Structured JSON -> Evaluates Importance Score & Criteria -> Persists QuizRankedConcept.
    """
    existing = quiz.ranked_concepts.all()
    if existing.exists() and not force:
        high_cnt = existing.filter(importance='HIGH').count()
        med_cnt = existing.filter(importance='MEDIUM').count()
        low_cnt = existing.filter(importance='LOW').count()
        return {
            'success': True,
            'already_ranked': True,
            'total_concepts': existing.count(),
            'high_count': high_cnt,
            'medium_count': med_cnt,
            'low_count': low_cnt,
        }

    # Ensure Phase 4 analysis is ready
    analysis = getattr(quiz, 'ai_analysis', None)
    if not analysis or analysis.status != 'completed':
        run_quiz_ai_analysis(quiz)
        analysis = getattr(quiz, 'ai_analysis', None)

    if not analysis or not analysis.structured_json:
        return {
            'success': False,
            'error': 'Phase 4 AI Content Analysis must be run before ranking concepts.'
        }

    data = analysis.structured_json
    api_key = get_api_key()

    ranked_items = []
    engine_name = "Noteforge Educational Evaluator (Deterministic Engine)"

    if api_key:
        try:
            # Prepare compact JSON payload of extracted entities
            compact_payload = json.dumps({
                'definitions': data.get('definitions', []),
                'important_concepts': data.get('important_concepts', []),
                'processes': data.get('processes', []),
                'formulas': data.get('formulas', []),
                'comparisons': data.get('comparisons', []),
                'applications': data.get('applications', []),
            }, indent=2)

            raw_ranked, engine_name = extract_with_gemini_ranking(compact_payload, api_key)
            if isinstance(raw_ranked, list) and len(raw_ranked) > 0:
                ranked_items = raw_ranked
        except Exception as e:
            logger.warning(f"LLM ranking failed, falling back to deterministic heuristic: {e}")
            ranked_items = fallback_deterministic_ranking(data, quiz)
            engine_name += " [Fallback after LLM error]"
    else:
        ranked_items = fallback_deterministic_ranking(data, quiz)

    # Clean & Persist QuizRankedConcept records
    records_to_create = []
    for item in ranked_items:
        concept_name = item.get('concept', '').strip()
        if not concept_name:
            continue

        importance = item.get('importance', 'MEDIUM').upper()
        if importance not in ['HIGH', 'MEDIUM', 'LOW']:
            importance = 'MEDIUM'

        score = item.get('importance_score', 50)
        try:
            score = int(score)
        except (ValueError, TypeError):
            score = 75 if importance == 'HIGH' else (55 if importance == 'MEDIUM' else 35)

        score = max(1, min(100, score))
        category = item.get('category', 'Core Concept')
        reason = item.get('reason', 'Educational concept identified in source material.')
        chunk_id = item.get('chunk_id', 'CHK-01')

        pages = item.get('pages', [1])
        start_p = min(pages) if (isinstance(pages, list) and pages) else 1
        end_p = max(pages) if (isinstance(pages, list) and pages) else start_p

        records_to_create.append(
            QuizRankedConcept(
                quiz=quiz,
                concept=concept_name[:250],
                category=category[:100],
                importance=importance,
                importance_score=score,
                reason=reason,
                chunk_id=chunk_id[:30],
                start_page=start_p,
                end_page=end_p,
                criteria_metadata=item.get('criteria', {}),
            )
        )

    QuizRankedConcept.objects.filter(quiz=quiz).delete()
    QuizRankedConcept.objects.bulk_create(records_to_create)

    high_cnt = sum(1 for r in records_to_create if r.importance == 'HIGH')
    med_cnt = sum(1 for r in records_to_create if r.importance == 'MEDIUM')
    low_cnt = sum(1 for r in records_to_create if r.importance == 'LOW')

    return {
        'success': True,
        'already_ranked': False,
        'engine': engine_name,
        'total_concepts': len(records_to_create),
        'high_count': high_cnt,
        'medium_count': med_cnt,
        'low_count': low_cnt,
    }
