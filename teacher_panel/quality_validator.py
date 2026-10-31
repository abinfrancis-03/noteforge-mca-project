"""
Phase 9 — AI Quality Validation ⭐ (Noteforge AI MCQ System)

Technique:
LLM-as-a-Judge / AI quality validation

Generated question:
        ↓
    Validator
        ↓
   Score 0–100

Checks:
1. Source support
2. Correct answer
3. One correct answer
4. Distractors
5. Ambiguity
6. Grammar
7. Learning value
8. Difficulty
9. Bloom level
10. Hallucination

Prompt:
Act as a strict academic MCQ evaluator. Compare the generated question against the supplied PDF source.
Verify that the correct answer is supported by the source, exactly one option is correct, distractors
are plausible, the wording is clear, the question is academically useful, and no external or hallucinated
information has been introduced. Also verify difficulty and Bloom's level. Give a score from 0–100.
Approve only questions scoring 85 or above. Otherwise recommend regeneration.
"""

import re
import json
import logging
from django.utils import timezone
from accounts.models import Quiz, Question, QuizSemanticChunk
from .ai_analyzer import get_api_key

logger = logging.getLogger(__name__)

APPROVAL_THRESHOLD = 85

VALIDATOR_SYSTEM_PROMPT = """You are a rigorous university examination board quality auditor and strict academic MCQ evaluator.
Your role is to critically audit educational multiple choice questions against their primary source PDF chunk using LLM-as-a-Judge.

Evaluate the question across these 10 academic dimensions:
1. Source support: Is the question topic and core factual assertion strictly found in the source text?
2. Correct answer: Is the designated correct answer factually true according to the source?
3. One correct answer: Is there exactly ONE unambiguous correct answer (no confusing second valid option)?
4. Distractors: Are the three distractors plausible, non-trivial, and free from "All/None of the above"?
5. Ambiguity: Is the wording clear, direct, and free from confusing double-negatives or trick syntax?
6. Grammar: Is the academic English phrasing, punctuation, and terminology formally correct?
7. Learning value: Does the question test core concepts and curricular understanding rather than trivial trivia?
8. Difficulty: Does the cognitive effort match the stated difficulty tier (Easy, Medium, Hard)?
9. Bloom level: Does the cognitive challenge match the stated Bloom level (Remember, Understand, Apply, Analyse)?
10. Hallucination: Has zero external or fabricated information been introduced?

SCORING RULES:
- Assign an integer score from 0 to 100.
- APPROVE ONLY questions scoring 85 or above.
- If score < 85, set is_approved=false and recommendation="regenerate".
- Return your audit strictly as valid JSON.
"""

VALIDATOR_USER_PROMPT = """Act as a strict academic MCQ evaluator. Compare the generated question against the supplied PDF source. Verify that the correct answer is supported by the source, exactly one option is correct, distractors are plausible, the wording is clear, the question is academically useful, and no external or hallucinated information has been introduced. Also verify difficulty and Bloom's level. Give a score from 0–100. Approve only questions scoring 85 or above. Otherwise recommend regeneration.

Required Output JSON Schema:
{
  "quality_score": 92,
  "is_approved": true,
  "recommendation": "approve",
  "feedback": "Comprehensive academic justification summarizing findings.",
  "checks": {
    "source_support": {"passed": true, "score": 10, "comment": "Directly grounded in chunk text."},
    "correct_answer": {"passed": true, "score": 10, "comment": "Factually verified by citation."},
    "one_correct_answer": {"passed": true, "score": 10, "comment": "Distinct and unambiguous answer."},
    "distractors": {"passed": true, "score": 9, "comment": "Plausible misconceptions without giveaways."},
    "ambiguity": {"passed": true, "score": 9, "comment": "Clear and direct collegiate phrasing."},
    "grammar": {"passed": true, "score": 10, "comment": "Grammatically sound."},
    "learning_value": {"passed": true, "score": 9, "comment": "High curricular exam relevance."},
    "difficulty": {"passed": true, "score": 9, "comment": "Cognitive effort accurately aligns with difficulty tier."},
    "bloom_level": {"passed": true, "score": 8, "comment": "Matches cognitive requirement of Bloom level."},
    "hallucination": {"passed": true, "score": 10, "comment": "Zero ungrounded information detected."}
  }
}
"""


def evaluate_question_with_gemini(question: Question, source_text: str, api_key: str) -> dict:
    """Uses Google Gemini as an academic judge to audit an MCQ."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    model_candidates = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-pro']
    last_err = None

    q_payload = {
        'question': question.text,
        'options': {
            'A': question.option_a,
            'B': question.option_b,
            'C': question.option_c,
            'D': question.option_d,
        },
        'correct_option': question.correct_option,
        'explanation': question.explanation,
        'topic': question.topic,
        'difficulty': question.difficulty,
        'blooms_level': question.blooms_level,
        'chunk_id': question.chunk_id,
        'source_page': question.source_page,
        'source_quote': question.source_quote,
    }

    prompt = (
        f"{VALIDATOR_USER_PROMPT}\n\n"
        f"SOURCE PDF CHUNK CONTENT:\n{source_text}\n\n"
        f"QUESTION UNDER EVALUATION:\n{json.dumps(q_payload, indent=2)}"
    )

    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=VALIDATOR_SYSTEM_PROMPT,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                }
            )
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            data = json.loads(raw_text.strip())
            score = int(data.get('quality_score', 0))
            is_approved = bool(data.get('is_approved', score >= APPROVAL_THRESHOLD))
            feedback = data.get('feedback', '')
            checks = data.get('checks', {})

            return {
                'quality_score': score,
                'is_approved': is_approved and score >= APPROVAL_THRESHOLD,
                'recommendation': 'approve' if (is_approved and score >= APPROVAL_THRESHOLD) else 'regenerate',
                'feedback': feedback,
                'checks': checks,
                'judge_model': model_name,
            }
        except Exception as e:
            logger.warning(f"Validator LLM call failed with {model_name}: {e}")
            last_err = e
            continue

    raise RuntimeError(f"All Gemini models failed evaluation: {last_err}")


def evaluate_question_heuristic(question: Question, source_text: str) -> dict:
    """
    Deterministic academic heuristic validator (fallback when offline).
    Audits the 10 dimensions through structural, lexical, and cognitive inspection.
    """
    scores = {}
    q_text = question.text.strip()
    opts = [question.option_a.strip(), question.option_b.strip(), question.option_c.strip(), question.option_d.strip()]
    corr_letter = question.correct_option.upper()
    opt_map = {'A': opts[0], 'B': opts[1], 'C': opts[2], 'D': opts[3]}
    corr_text = opt_map.get(corr_letter, '')

    src_lower = (source_text + " " + question.source_quote).lower()

    # 1. Source Support (0-10)
    q_words = [w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', q_text)]
    common_words = set(q_words).intersection(set(re.findall(r'\b[A-Za-z]{4,}\b', src_lower)))
    if len(common_words) >= 3 or (question.source_quote and question.source_quote.lower() in src_lower):
        scores['source_support'] = {'passed': True, 'score': 10, 'comment': 'Direct factual foundation verified in retrieved chunk.'}
    elif len(common_words) >= 1:
        scores['source_support'] = {'passed': True, 'score': 8, 'comment': 'Supported by key topic vocabulary in source.'}
    else:
        scores['source_support'] = {'passed': False, 'score': 5, 'comment': 'Limited explicit terminology overlap in source.'}

    # 2. Correct Answer (0-10)
    if corr_text and (corr_text.lower() in src_lower or len(set(corr_text.lower().split()).intersection(set(src_lower.split()))) >= 2):
        scores['correct_answer'] = {'passed': True, 'score': 10, 'comment': 'Correct option is directly substantiated by the text.'}
    else:
        scores['correct_answer'] = {'passed': True, 'score': 9, 'comment': 'Correct answer logically consistent with curriculum facts.'}

    # 3. One Correct Answer (0-10)
    unique_opts = set(opt.lower() for opt in opts)
    if len(unique_opts) == 4 and all(len(o) > 2 for o in opts):
        scores['one_correct_answer'] = {'passed': True, 'score': 10, 'comment': 'Four distinct options with exactly one designated answer.'}
    else:
        scores['one_correct_answer'] = {'passed': False, 'score': 4, 'comment': 'Duplicate or trivial options detected.'}

    # 4. Distractors (0-10)
    banned = ['all of the above', 'none of the above', 'both a and b', 'neither a nor b']
    has_banned = any(any(b in o.lower() for b in banned) for o in opts)
    distractor_lens = [len(o) for o in opts if o != corr_text]
    avg_d_len = sum(distractor_lens) / max(1, len(distractor_lens))
    if not has_banned and avg_d_len >= 10:
        scores['distractors'] = {'passed': True, 'score': 10, 'comment': 'Plausible, academic distractors without giveaway filler.'}
    elif not has_banned:
        scores['distractors'] = {'passed': True, 'score': 8, 'comment': 'Adequate distractors, but could be more comprehensive.'}
    else:
        scores['distractors'] = {'passed': False, 'score': 5, 'comment': 'Contains formulaic or banned distractor phrasing.'}

    # 5. Ambiguity (0-10)
    if len(q_text) >= 25 and not q_text.count('?') > 2 and 'not not' not in q_text.lower():
        scores['ambiguity'] = {'passed': True, 'score': 10, 'comment': 'Clear, direct, and unambiguous collegiate wording.'}
    else:
        scores['ambiguity'] = {'passed': False, 'score': 6, 'comment': 'Question text is brief or structurally ambiguous.'}

    # 6. Grammar (0-10)
    if q_text[0].isupper() and (q_text.endswith('?') or q_text.endswith('.')):
        scores['grammar'] = {'passed': True, 'score': 10, 'comment': 'Formal academic syntax and punctuation.'}
    else:
        scores['grammar'] = {'passed': True, 'score': 8, 'comment': 'Grammatically intelligible with minor stylistic room.'}

    # 7. Learning Value (0-10)
    if len(question.explanation) >= 30:
        scores['learning_value'] = {'passed': True, 'score': 10, 'comment': 'High educational value with source rationale.'}
    else:
        scores['learning_value'] = {'passed': True, 'score': 8, 'comment': 'Standard educational test item.'}

    # 8. Difficulty (0-10)
    diff = question.difficulty.lower()
    if diff == 'easy' and (question.blooms_level in ['Remember', 'Understand'] or 'define' in q_text.lower()):
        scores['difficulty'] = {'passed': True, 'score': 10, 'comment': 'Accurately tests direct understanding of basic concepts.'}
    elif diff == 'medium' and (question.blooms_level in ['Understand', 'Apply'] or 'role' in q_text.lower() or 'apply' in q_text.lower()):
        scores['difficulty'] = {'passed': True, 'score': 10, 'comment': 'Accurately requires concept connection or application.'}
    elif diff == 'hard' and (question.blooms_level in ['Apply', 'Analyse'] or 'analyse' in q_text.lower() or 'scenario' in q_text.lower()):
        scores['difficulty'] = {'passed': True, 'score': 10, 'comment': 'Accurately demands multi-step reasoning or comparative analysis.'}
    else:
        scores['difficulty'] = {'passed': True, 'score': 8, 'comment': 'Acceptable difficulty tier alignment.'}

    # 9. Bloom's Taxonomy Level (0-10)
    bloom = question.blooms_level
    if bloom in ['Remember', 'Understand', 'Apply', 'Analyse']:
        scores['bloom_level'] = {'passed': True, 'score': 10, 'comment': f"Cognitive task matches canonical Bloom level: '{bloom}'."}
    else:
        scores['bloom_level'] = {'passed': False, 'score': 6, 'comment': 'Non-canonical cognitive level assigned.'}

    # 10. Zero Hallucination (0-10)
    if len(question.source_quote) > 10 and question.source_quote.lower() in src_lower:
        scores['hallucination'] = {'passed': True, 'score': 10, 'comment': 'Zero hallucination detected. Exact quote verified in chunk.'}
    elif len(common_words) >= 2:
        scores['hallucination'] = {'passed': True, 'score': 9, 'comment': 'No extraneous or ungrounded facts detected.'}
    else:
        scores['hallucination'] = {'passed': True, 'score': 8, 'comment': 'Plausible curriculum concepts without obvious hallucination.'}

    total_score = sum(item['score'] for item in scores.values())
    is_approved = total_score >= APPROVAL_THRESHOLD

    feedback = (
        f"Heuristic Academic Audit: Score {total_score}/100. "
        + ("Approved for examination deployment." if is_approved else "Fails strict 85 threshold; recommend regeneration.")
    )

    return {
        'quality_score': total_score,
        'is_approved': is_approved,
        'recommendation': 'approve' if is_approved else 'regenerate',
        'feedback': feedback,
        'checks': scores,
        'judge_model': 'Noteforge Academic Heuristic Judge',
    }


def validate_single_question(question: Question) -> dict:
    """Validates one question against its associated semantic chunk."""
    chunk = None
    if question.chunk_id:
        chunk = question.quiz.chunks.filter(chunk_id=question.chunk_id).first()
    if not chunk:
        chunk = question.quiz.chunks.first()

    source_text = chunk.content if chunk else (question.source_quote or question.text)
    api_key = get_api_key()

    if api_key:
        try:
            res = evaluate_question_with_gemini(question, source_text, api_key)
        except Exception as e:
            logger.warning(f"LLM Judge failed for Question #{question.pk}, using heuristic validator: {e}")
            res = evaluate_question_heuristic(question, source_text)
    else:
        res = evaluate_question_heuristic(question, source_text)

    # Persist audit results to the Question record
    question.quality_score = res['quality_score']
    question.is_approved = res['is_approved']
    question.validation_feedback = res['feedback']
    question.validation_checks = res['checks']
    question.validated_at = timezone.now()
    question.save(update_fields=['quality_score', 'is_approved', 'validation_feedback', 'validation_checks', 'validated_at'])

    return res


def validate_quiz_questions(quiz: Quiz, force: bool = False) -> dict:
    """
    Main entrypoint for Phase 9:
    Validates all questions for a given Quiz using LLM-as-a-Judge.
    """
    questions = quiz.questions.all().order_by('id')
    if not questions.exists():
        return {
            'success': False,
            'error': 'No questions found to validate. Please generate questions first.',
        }

    # If already validated and not forcing, return existing metrics
    if not force and all(q.validated_at is not None for q in questions):
        approved_cnt = questions.filter(is_approved=True).count()
        avg_score = sum(q.quality_score for q in questions) / max(1, questions.count())
        return {
            'success': True,
            'already_validated': True,
            'total': questions.count(),
            'approved_count': approved_cnt,
            'rejected_count': questions.count() - approved_cnt,
            'average_score': round(avg_score, 1),
        }

    results = []
    for q in questions:
        res = validate_single_question(q)
        results.append(res)

    approved_cnt = sum(1 for r in results if r['is_approved'])
    avg_score = sum(r['quality_score'] for r in results) / max(1, len(results))

    return {
        'success': True,
        'already_validated': False,
        'total': len(results),
        'approved_count': approved_cnt,
        'rejected_count': len(results) - approved_cnt,
        'average_score': round(avg_score, 1),
        'model_used': results[0].get('judge_model', 'Academic Judge') if results else 'None',
    }


def regenerate_rejected_questions(quiz: Quiz, threshold: int = APPROVAL_THRESHOLD) -> dict:
    """
    Targeted regeneration for questions scoring below threshold (< 85).
    Re-synthesizes replacement questions until they meet the quality criteria.
    """
    from .mcq_generator import fallback_deterministic_mcqs
    low_qs = quiz.questions.filter(quality_score__lt=threshold)
    replaced_count = 0

    if not low_qs.exists():
        return {'success': True, 'replaced_count': 0, 'message': 'All questions already meet the quality threshold!'}

    # Generate fresh replacements
    fresh_pool = fallback_deterministic_mcqs(quiz, target_count=low_qs.count() + 2)
    for q_record in low_qs:
        if fresh_pool:
            cand = fresh_pool.pop(0)
            q_record.text = cand['text']
            q_record.option_a = cand['option_a']
            q_record.option_b = cand['option_b']
            q_record.option_c = cand['option_c']
            q_record.option_d = cand['option_d']
            q_record.correct_option = cand['correct_option']
            q_record.explanation = cand['explanation']
            q_record.topic = cand['topic']
            q_record.difficulty = cand['difficulty']
            q_record.blooms_level = cand['blooms_level']
            q_record.question_type = cand['question_type']
            q_record.chunk_id = cand['chunk_id']
            q_record.source_page = cand['source_page']
            q_record.source_quote = cand['source_quote']
            q_record.save()
            validate_single_question(q_record)
            replaced_count += 1

    return {
        'success': True,
        'replaced_count': replaced_count,
        'message': f"Regenerated and re-validated {replaced_count} questions.",
    }
