"""
Phase 14 — Performance Analytics (Noteforge AI MCQ System)

Technique:
Topic-wise learning analytics

Instead of only:
You scored 14/20.

Show:
Normalization             90%
Database Keys             80%
Functional Dependency     45%  ← Weak
Transactions              85%

Prompt:
Analyse the student's quiz results using question topic metadata. Calculate overall score and topic-wise performance.
Identify weak topics based on low performance and recommend which topics the student should revise.
Do not make personal or psychological judgments.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from django.conf import settings
from django.utils import timezone
from accounts.models import Quiz, QuizAttempt, StudentAnswer, Question

logger = logging.getLogger(__name__)

# Performance thresholds
WEAK_THRESHOLD = 60.0    # Accuracy below 60% indicates weak topic / needs revision
STRONG_THRESHOLD = 80.0  # Accuracy >= 80% indicates strong mastery / proficient

ANALYTICS_SYSTEM_PROMPT = """You are an objective academic learning analytics advisor for university students.
Your role is to analyze quiz assessment results strictly using question topic metadata and provide concept-focused revision guidance.

MANDATORY GUIDELINES:
1. Do NOT make personal or psychological judgments (e.g., do NOT comment on intelligence, aptitude, attention, mood, laziness, or emotional states).
2. Focus exclusively on academic subject matter, specific concepts, rules, theorems, processes, and textbook definitions.
3. Identify weak topics where performance is below 60% and clearly recommend actionable topics and study points to revise.
4. Return structured JSON with clear educational recommendations grounded in the assessed material.
"""

ANALYTICS_USER_PROMPT = """Analyse the student's quiz results using question topic metadata. Calculate overall score and topic-wise performance. Identify weak topics based on low performance and recommend which topics the student should revise. Do not make personal or psychological judgments.

Assess the provided topic breakdown and question response metadata, and output a valid JSON object matching this schema:
{
  "overall_assessment": "Direct objective summary of academic performance across assessed topics",
  "weak_topics_identified": [
    {
      "topic": "Topic Name",
      "accuracy": 45.0,
      "priority": "High",
      "concepts_to_review": ["Key concept 1", "Key concept 2"],
      "revision_guidance": "Specific academic study advice for this topic"
    }
  ],
  "recommended_revision_actions": [
    "Concrete revision action 1",
    "Concrete revision action 2"
  ],
  "academic_study_plan": "Concise next-step revision strategy focused purely on course materials"
}
"""


def get_gemini_api_key() -> str:
    """Retrieves Google Gemini API key from environment or settings."""
    try:
        from teacher_panel.ai_analyzer import get_api_key
        key = get_api_key()
        if key:
            return key
    except Exception:
        pass
    return (
        os.getenv('GEMINI_API_KEY', '') or
        getattr(settings, 'GEMINI_API_KEY', '') or
        getattr(settings, 'GOOGLE_API_KEY', '')
    )


def calculate_topic_breakdown(attempt: QuizAttempt) -> Dict[str, Any]:
    """
    Computes topic-wise performance analytics from student answers.
    Calculates overall score, question counts, marks, accuracy percentage,
    and identifies weak topics (< 60%) vs proficient vs strong topics.
    """
    student_answers = attempt.student_answers.select_related('question').all()
    
    topic_map: Dict[str, Dict[str, Any]] = {}

    if student_answers.exists():
        for sa in student_answers:
            q = sa.question
            topic_name = (q.topic.strip() if q.topic else '').strip() or 'General Concepts'
            
            if topic_name not in topic_map:
                topic_map[topic_name] = {
                    'topic': topic_name,
                    'total_questions': 0,
                    'correct_count': 0,
                    'incorrect_count': 0,
                    'marks_obtained': 0,
                    'total_marks': 0,
                    'questions': [],
                }
            
            topic_map[topic_name]['total_questions'] += 1
            topic_map[topic_name]['total_marks'] += q.marks
            if sa.is_correct:
                topic_map[topic_name]['correct_count'] += 1
                topic_map[topic_name]['marks_obtained'] += sa.marks_obtained
            else:
                topic_map[topic_name]['incorrect_count'] += 1

            topic_map[topic_name]['questions'].append({
                'id': q.id,
                'text': q.text,
                'selected_option': sa.selected_option,
                'correct_option': q.correct_option,
                'is_correct': sa.is_correct,
                'explanation': q.explanation,
                'difficulty': q.difficulty,
                'blooms_level': q.blooms_level,
                'source_page': q.source_page,
            })
    else:
        # Fallback to answers_json if StudentAnswer records are not populated
        answers_json = attempt.answers_json or {}
        for q_pk, q_data in answers_json.items():
            topic_name = (q_data.get('topic') or '').strip() or 'General Concepts'
            if topic_name not in topic_map:
                topic_map[topic_name] = {
                    'topic': topic_name,
                    'total_questions': 0,
                    'correct_count': 0,
                    'incorrect_count': 0,
                    'marks_obtained': 0,
                    'total_marks': 0,
                    'questions': [],
                }
            topic_map[topic_name]['total_questions'] += 1
            topic_map[topic_name]['total_marks'] += 1
            is_corr = q_data.get('is_correct', False)
            if is_corr:
                topic_map[topic_name]['correct_count'] += 1
                topic_map[topic_name]['marks_obtained'] += 1
            else:
                topic_map[topic_name]['incorrect_count'] += 1

            topic_map[topic_name]['questions'].append({
                'id': q_pk,
                'text': q_data.get('question_text', ''),
                'selected_option': q_data.get('chosen_option', ''),
                'correct_option': q_data.get('correct_option', ''),
                'is_correct': is_corr,
                'explanation': q_data.get('explanation', ''),
                'difficulty': q_data.get('difficulty', 'medium'),
                'blooms_level': q_data.get('blooms_level', 'Understand'),
                'source_page': 1,
            })

    # Compute percentages and classification for each topic
    topics_list: List[Dict[str, Any]] = []
    weak_topics: List[str] = []
    strong_topics: List[str] = []
    proficient_topics: List[str] = []

    for t_name, data in topic_map.items():
        total = data['total_questions']
        correct = data['correct_count']
        pct = round((correct / total * 100), 1) if total > 0 else 0.0
        is_weak = (pct < WEAK_THRESHOLD)

        if is_weak:
            status = 'weak'
            status_label = 'Weak'
            badge_class = 'badge-red'
            weak_topics.append(t_name)
        elif pct >= STRONG_THRESHOLD:
            status = 'strong'
            status_label = 'Mastered'
            badge_class = 'badge-green'
            strong_topics.append(t_name)
        else:
            status = 'proficient'
            status_label = 'Proficient'
            badge_class = 'badge-blue'
            proficient_topics.append(t_name)

        data['accuracy'] = pct
        data['percentage'] = pct
        data['is_weak'] = is_weak
        data['status'] = status
        data['status_label'] = status_label
        data['badge_class'] = badge_class
        topics_list.append(data)

    # Sort topics: weak topics first (< 60%), then by ascending accuracy
    topics_list.sort(key=lambda t: (not t['is_weak'], t['percentage']))

    return {
        'overall_score': attempt.score,
        'total_questions': attempt.total_questions,
        'percentage': attempt.percentage,
        'time_taken_seconds': attempt.time_taken_seconds,
        'attempt_number': attempt.attempt_number,
        'total_topics': len(topics_list),
        'weak_topics_count': len(weak_topics),
        'strong_topics_count': len(strong_topics),
        'proficient_topics_count': len(proficient_topics),
        'weak_topics': weak_topics,
        'strong_topics': strong_topics,
        'proficient_topics': proficient_topics,
        'topics': topics_list,
    }


def generate_deterministic_recommendations(breakdown: Dict[str, Any], quiz_title: str) -> Dict[str, Any]:
    """
    Deterministic academic recommendation engine.
    Strictly follows: 'Do not make personal or psychological judgments'.
    Generates actionable revision recommendations based on weak topics and source question explanations.
    """
    weak_topics = breakdown.get('weak_topics', [])
    topics = breakdown.get('topics', [])
    overall_pct = breakdown.get('percentage', 0.0)

    weak_topics_identified = []
    recommended_actions = []

    for t in topics:
        if t.get('is_weak', False):
            t_name = t['topic']
            t_pct = t['percentage']
            # Find explanations for missed questions in this topic
            missed_exps = []
            for q in t.get('questions', []):
                if not q.get('is_correct') and q.get('explanation'):
                    clean_exp = q['explanation'].strip()
                    if clean_exp and clean_exp not in missed_exps:
                        missed_exps.append(clean_exp)

            concepts = []
            for exp in missed_exps[:3]:
                # Extract core clause as concept summary
                first_sentence = exp.split('.')[0].strip()
                if first_sentence:
                    concepts.append(first_sentence)

            if not concepts:
                concepts = [f"Core definitions and theorems of {t_name}", f"Standard applications and problem solving in {t_name}"]

            guidance = f"Review textbook sections covering {t_name}. Focus particularly on foundational principles and distinction between related definitions."
            
            weak_topics_identified.append({
                'topic': t_name,
                'accuracy': t_pct,
                'priority': 'High' if t_pct < 40 else 'Medium',
                'concepts_to_review': concepts,
                'revision_guidance': guidance,
            })

            recommended_actions.append(
                f"Revise {t_name} (Accuracy: {t_pct}%): Re-read primary definitions and work through solved examples."
            )

    if not weak_topics_identified:
        summary = f"All assessed topics meet or exceed the 60% proficiency threshold on '{quiz_title}' ({overall_pct}% overall). High conceptual understanding demonstrated."
        actions = ["Maintain mastery by reviewing synthesis and multi-step reasoning problems across all topics."]
        study_plan = "Proceed to advanced practice problems and inter-topic comparison questions."
    else:
        topic_names_str = ", ".join([f"'{w['topic']}' ({w['accuracy']}%)" for w in weak_topics_identified])
        summary = f"Assessment indicates {len(weak_topics_identified)} topic(s) below the 60% mastery threshold: {topic_names_str}. Targeted revision is recommended prior to retesting."
        actions = recommended_actions
        study_plan = f"Allocate primary study time to {weak_topics_identified[0]['topic']} before re-attempting the assessment."

    return {
        'overall_assessment': summary,
        'weak_topics_identified': weak_topics_identified,
        'recommended_revision_actions': actions,
        'academic_study_plan': study_plan,
        'generated_by': 'deterministic_engine',
    }


def generate_ai_recommendations(breakdown: Dict[str, Any], quiz_title: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Invokes Google Gemini with the exact Phase 14 prompt to produce topic-wise learning recommendations.
    Enforces zero personal/psychological judgments.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model_candidates = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-pro']
        
        # Build context payload
        weak_topics_data = [
            {
                'topic': t['topic'],
                'accuracy': f"{t['percentage']}%",
                'total_questions': t['total_questions'],
                'correct': t['correct_count'],
                'missed_questions': [
                    {'text': q['text'], 'explanation': q['explanation']}
                    for q in t.get('questions', []) if not q.get('is_correct')
                ][:3]
            }
            for t in breakdown.get('topics', []) if t.get('is_weak')
        ]

        all_topics_summary = [
            {'topic': t['topic'], 'accuracy': f"{t['percentage']}%", 'is_weak': t['is_weak']}
            for t in breakdown.get('topics', [])
        ]

        payload = {
            'quiz_title': quiz_title,
            'overall_score': f"{breakdown['overall_score']}/{breakdown['total_questions']}",
            'overall_percentage': f"{breakdown['percentage']}%",
            'all_topics_performance': all_topics_summary,
            'weak_topics_requiring_revision': weak_topics_data,
        }

        full_prompt = (
            f"{ANALYTICS_USER_PROMPT}\n\n"
            f"STUDENT ASSESSMENT DATA:\n{json.dumps(payload, indent=2)}"
        )

        for model_name in model_candidates:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=ANALYTICS_SYSTEM_PROMPT,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                    }
                )
                response = model.generate_content(full_prompt)
                raw_text = response.text.strip()

                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                data = json.loads(raw_text.strip())
                data['generated_by'] = f'ai_{model_name}'
                return data
            except Exception as model_err:
                logger.warning(f"AI recommendation model '{model_name}' failed: {model_err}")
                continue

    except Exception as e:
        logger.error(f"Error in generate_ai_recommendations: {e}")

    return None


def run_topic_performance_analytics(attempt: QuizAttempt, force_refresh: bool = False, use_ai: bool = True) -> Dict[str, Any]:
    """
    Main entrypoint for Phase 14 Performance Analytics.
    1. Calculates topic breakdown from student response metadata.
    2. Identifies weak topics (< 60%).
    3. Generates educational revision recommendations (AI or deterministic).
    4. Persists the structured analytics in attempt.topic_analysis.
    """
    if not force_refresh and attempt.topic_analysis and 'topics' in attempt.topic_analysis:
        return attempt.topic_analysis

    quiz_title = attempt.quiz.title if attempt.quiz else "Quiz Assessment"
    breakdown = calculate_topic_breakdown(attempt)

    recommendations = None
    if use_ai and breakdown.get('weak_topics_count', 0) > 0:
        api_key = get_gemini_api_key()
        if api_key:
            recommendations = generate_ai_recommendations(breakdown, quiz_title, api_key)

    if not recommendations:
        recommendations = generate_deterministic_recommendations(breakdown, quiz_title)

    breakdown['recommendations'] = recommendations
    breakdown['analyzed_at'] = timezone.now().isoformat()

    # Persist in QuizAttempt
    attempt.topic_analysis = breakdown
    attempt.save(update_fields=['topic_analysis'])

    return breakdown


def get_student_cumulative_topic_analytics(student, subject=None) -> Dict[str, Any]:
    """
    Aggregates topic-wise learning analytics across all quiz attempts for a student.
    Provides lifetime mastery status, identifying persisting weak topics and strengths.
    """
    attempts_qs = QuizAttempt.objects.filter(student=student).select_related('quiz', 'quiz__subject')
    if subject:
        attempts_qs = attempts_qs.filter(quiz__subject=subject)

    total_attempts = attempts_qs.count()
    if total_attempts == 0:
        return {
            'total_attempts': 0,
            'topics': [],
            'weak_topics': [],
            'strong_topics': [],
            'overall_avg_pct': 0.0,
        }

    topic_totals: Dict[str, Dict[str, Any]] = {}

    for attempt in attempts_qs:
        # Use existing topic_analysis or calculate on-the-fly
        data = attempt.topic_analysis
        if not data or 'topics' not in data:
            data = calculate_topic_breakdown(attempt)

        for t in data.get('topics', []):
            t_name = t['topic']
            if t_name not in topic_totals:
                topic_totals[t_name] = {
                    'topic': t_name,
                    'total_questions': 0,
                    'correct_count': 0,
                    'quizzes_count': 0,
                }
            topic_totals[t_name]['total_questions'] += t['total_questions']
            topic_totals[t_name]['correct_count'] += t['correct_count']
            topic_totals[t_name]['quizzes_count'] += 1

    cumulative_topics = []
    weak_topics = []
    strong_topics = []
    total_q_all = 0
    correct_q_all = 0

    for t_name, data in topic_totals.items():
        tot = data['total_questions']
        corr = data['correct_count']
        pct = round((corr / tot * 100), 1) if tot > 0 else 0.0
        is_weak = (pct < WEAK_THRESHOLD)

        total_q_all += tot
        correct_q_all += corr

        if is_weak:
            status = 'weak'
            status_label = 'Weak'
            badge_class = 'badge-red'
            weak_topics.append(t_name)
        elif pct >= STRONG_THRESHOLD:
            status = 'strong'
            status_label = 'Mastered'
            badge_class = 'badge-green'
            strong_topics.append(t_name)
        else:
            status = 'proficient'
            status_label = 'Proficient'
            badge_class = 'badge-blue'
            cumulative_topics.append(t_name)

        data['accuracy'] = pct
        data['percentage'] = pct
        data['is_weak'] = is_weak
        data['status'] = status
        data['status_label'] = status_label
        data['badge_class'] = badge_class
        cumulative_topics.append(data)

    cumulative_topics.sort(key=lambda t: (not t['is_weak'], t['percentage']))
    overall_avg = round((correct_q_all / total_q_all * 100), 1) if total_q_all > 0 else 0.0

    return {
        'total_attempts': total_attempts,
        'total_questions_answered': total_q_all,
        'overall_avg_pct': overall_avg,
        'topics': cumulative_topics,
        'weak_topics': weak_topics,
        'strong_topics': strong_topics,
        'weak_count': len(weak_topics),
        'strong_count': len(strong_topics),
    }
