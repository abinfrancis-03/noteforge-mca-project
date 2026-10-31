"""
Source-Grounded RAG-based MCQ Generation Engine (Phase 6 — Noteforge AI MCQ System)

Technique:
Source-grounded RAG-based generation

Methods:
Retrieved PDF chunks
        +
Important concepts (HIGH/MEDIUM)
        +
Teacher settings (num_questions, difficulty, question_types)
        ↓
       LLM
        ↓
      MCQs

Prompt:
Generate college-level MCQs for Noteforge using only the supplied retrieved PDF content and extracted concepts.
Use source-grounded generation. Every question must be supported by the source material. Generate exactly four
options and exactly one correct answer. Create conceptual, understanding, application, scenario-based and
analytical questions where supported by the material. Use plausible distractors based on related concepts or
common misconceptions. Do not create trivial, ambiguous, duplicate or unsupported questions. Return the
question, four options, correct answer, explanation, topic, difficulty, Bloom's level and source page/chunk as JSON.
"""

import os
import re
import json
import random
import logging
from django.conf import settings
from accounts.models import Quiz, Question, QuizSemanticChunk, QuizRankedConcept
from .ai_analyzer import get_api_key, run_quiz_ai_analysis
from .concept_ranker import rank_quiz_concepts

logger = logging.getLogger(__name__)

BLOOMS_DISTRIBUTION_TARGETS = {
    'Remember': 0.20,
    'Understand': 0.30,
    'Apply': 0.35,
    'Analyse': 0.15,
}

DIFFICULTY_DISTRIBUTION_TARGETS = {
    'easy': 0.25,
    'medium': 0.50,
    'hard': 0.25,
}


def calculate_difficulty_quotas(total_questions: int, difficulty_setting: str = 'mixed') -> dict:
    """
    Calculates difficulty quotas according to Phase 8 specifications:
    - Easy: direct understanding
    - Medium: concept connection/application
    - Hard: reasoning/analysis
    For mixed: 25% Easy, 50% Medium, 25% Hard.
    For pure settings ('easy', 'medium', 'hard'): 100% of requested tier.
    """
    if total_questions <= 0:
        return {'easy': 0, 'medium': 0, 'hard': 0}

    setting = (difficulty_setting or 'mixed').lower().strip()
    if setting == 'easy':
        return {'easy': total_questions, 'medium': 0, 'hard': 0}
    elif setting == 'medium':
        return {'easy': 0, 'medium': total_questions, 'hard': 0}
    elif setting == 'hard':
        return {'easy': 0, 'medium': 0, 'hard': total_questions}

    # Mixed distribution: 25% Easy, 50% Medium, 25% Hard
    easy_q = max(1, round(total_questions * 0.25)) if total_questions >= 4 else 1
    hard_q = max(1, round(total_questions * 0.25)) if total_questions >= 4 else (1 if total_questions >= 3 else 0)
    med_q = total_questions - (easy_q + hard_q)
    if med_q < 1 and total_questions >= 2:
        med_q = 1
        if easy_q > 1:
            easy_q -= 1
        elif hard_q > 1:
            hard_q -= 1

    current_sum = easy_q + med_q + hard_q
    diff = total_questions - current_sum
    if diff > 0:
        med_q += diff
    elif diff < 0:
        for _ in range(abs(diff)):
            if med_q > 1:
                med_q -= 1
            elif hard_q > 1:
                hard_q -= 1
            elif easy_q > 1:
                easy_q -= 1

    return {
        'easy': easy_q,
        'medium': med_q,
        'hard': hard_q,
    }


def calculate_blooms_quotas(total_questions: int) -> dict:
    """
    Calculates target quotas according to the pedagogical distribution:
    20% Remember, 30% Understand, 35% Apply, 15% Analyse.
    """
    rem = max(1, round(total_questions * 0.20)) if total_questions >= 5 else 1
    und = max(1, round(total_questions * 0.30)) if total_questions >= 4 else 1
    app = max(1, round(total_questions * 0.35)) if total_questions >= 3 else 1
    ana = total_questions - (rem + und + app)
    if ana < 1 and total_questions >= 6:
        ana = 1
        if app > 1:
            app -= 1
        elif und > 1:
            und -= 1
    elif ana < 0:
        ana = 0

    current_sum = rem + und + app + ana
    diff = total_questions - current_sum
    if diff > 0:
        app += diff
    elif diff < 0:
        for _ in range(abs(diff)):
            if app > 1:
                app -= 1
            elif und > 1:
                und -= 1
            elif rem > 1:
                rem -= 1

    return {
        'Remember': rem,
        'Understand': und,
        'Apply': app,
        'Analyse': ana,
    }


MCQ_SYSTEM_PROMPT = """You are an experienced, student-centric college educator creating an exam revision quiz for Noteforge.
Your mission is to generate clear, simple, and highly useful multiple-choice questions (MCQs) based strictly on the provided study material.

CORE PRINCIPLES FOR STUDENT USEFULNESS:
1. Simplicity & Clarity: Write concise, unambiguous questions. Avoid overly complex, convoluted phrasing, double negatives, or trick questions. Questions must be easy for students to understand.
2. High-Yield & Useful: Focus on fundamental concepts, key definitions, primary advantages, core principles, and practical scenarios that a student must master for their exams.
3. Relevant, Plausible Options:
   - Every question must have exactly 4 clean options (option_a, option_b, option_c, option_d) and exactly ONE correct answer.
   - Use other real concepts or terms from the same study material as distractors (e.g., if the question is about Inheritance, use Encapsulation, Polymorphism, and Abstraction as distractors).
   - Never use "All of the above", "None of the above", or nonsensical filler.
4. Student-Friendly Explanations:
   - Provide a clear, encouraging 1–2 sentence explanation explaining why the correct answer is right and highlighting the core concept to help students learn and revise.
5. Strict Grounding: Use ONLY the provided retrieved content from the teacher's uploaded materials.
6. No Meta-Phrases: NEVER start or prefix questions with phrases like 'According to the study material', 'Based on the text', 'In the provided notes', or 'As stated in the document'. Just ask the question directly.
"""

MCQ_USER_PROMPT = """Generate simple, clear, and high-yield multiple-choice questions (MCQs) that are most useful for students to understand and master the key concepts in the material.
Ensure questions have clear wording, plausible subject-related distractors, and a helpful explanation.
Do NOT use meta-phrases like 'According to the study material, ' in any question.

Required Output Schema:
[
  {
    "text": "Clear, direct question for the student?",
    "option_a": "Option A text",
    "option_b": "Option B text",
    "option_c": "Option C text",
    "option_d": "Option D text",
    "correct_option": "A | B | C | D",
    "explanation": "Clear, helpful explanation reinforcing the key concept for the student.",
    "topic": "Topic Name",
    "difficulty": "easy | medium | hard",
    "blooms_level": "Remember | Understand | Apply | Analyse",
    "question_type": "conceptual | understanding | application",
    "chunk_id": "CHK-01",
    "source_page": 1,
    "source_quote": "Direct quote from source supporting the question."
  }
]
"""


META_PATTERNS = [
    r"^(?:according to|based on|as per|as stated in|in)\s+(?:the\s+|this\s+|the provided\s+)?(?:study material|provided (?:text|document|material)|source (?:text|material|curriculum)|text|document|notes|material|chapter|module\s*\w+)[^,:\?]*[,:]\s*",
    r",?\s*\b(?:according to|as stated in|as per)\s+(?:the\s+|this\s+|the provided\s+)?(?:study material|source material|document|notes|provided text)\b",
]

def clean_meta_question_text(text: str) -> str:
    """Strips meta-references like 'According to the study material, ' from question or explanation."""
    q_str = (text or '').strip()
    for pat in META_PATTERNS:
        q_str = re.sub(pat, '', q_str, flags=re.IGNORECASE).strip()
    if q_str:
        q_str = q_str[0].upper() + q_str[1:]
    return q_str


def format_rag_context(quiz: Quiz) -> str:
    """Formats retrieved semantic chunks and prioritized concepts into a unified RAG context block."""
    chunks = quiz.chunks.all().order_by('chunk_index')
    ranked_concepts = quiz.ranked_concepts.filter(importance__in=['HIGH', 'MEDIUM']).order_by('-importance_score')

    lines = []
    lines.append(f"TEACHER ASSESSMENT CONFIGURATION:")
    lines.append(f"- Number of Questions Requested: {quiz.num_questions}")
    lines.append(f"- Difficulty Level: {quiz.difficulty}")
    lines.append(f"- Question Types Preference: {quiz.question_types}")
    lines.append(f"- Subject: {quiz.subject.name} ({quiz.subject.subject_code})\n")

    lines.append("=== RETRIEVED HIGH-PRIORITY CONCEPTS ===")
    for rc in ranked_concepts[:12]:
        lines.append(f"• [{rc.importance} - {rc.importance_score}] {rc.concept} ({rc.category}) | Source: {rc.chunk_id}, p.{rc.start_page}")
        lines.append(f"  Rationale: {rc.reason}")

    lines.append("\n=== RETRIEVED PDF CONTENT CHUNKS ===")
    for c in chunks:
        lines.append(f"--- CHUNK {c.chunk_id} (Pages {c.start_page}–{c.end_page}) ---")
        lines.append(f"Topic: {c.topic}")
        lines.append(f"Content:\n{c.content.strip()}\n")

    return "\n".join(lines)


def generate_mcqs_with_gemini(rag_context: str, api_key: str, target_count: int) -> tuple[list, str]:
    """Uses Google Gemini with RAG context to synthesize college-level MCQs."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model_candidates = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-pro']
    last_err = None

    prompt = (
        f"{MCQ_USER_PROMPT}\n\n"
        f"Generate exactly {target_count} multiple choice questions.\n\n"
        f"RAG CONTEXT:\n{rag_context}"
    )

    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=MCQ_SYSTEM_PROMPT,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.3,
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

            parsed = json.loads(raw_text.strip())
            if isinstance(parsed, dict):
                for k in ['questions', 'mcqs', 'items']:
                    if k in parsed and isinstance(parsed[k], list):
                        parsed = parsed[k]
                        break

            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed, model_name

        except Exception as e:
            logger.warning(f"MCQ generation failed with model {model_name}: {e}")
            last_err = e
            continue

    raise RuntimeError(f"All Gemini models failed MCQ generation. Last error: {last_err}")


def fallback_deterministic_mcqs(quiz: Quiz, target_count: int) -> list:
    """
    High-yield student-centric deterministic MCQ generation engine.
    Extracts core concepts, definitions, advantages, and principles directly from
    the study material and creates clear, direct, and exam-relevant MCQs.
    """
    import random
    subject_name = (quiz.subject.name if quiz.subject else 'Course Curriculum').strip()
    
    def clean_term(term_str):
        if not term_str:
            return ""
        t = re.sub(r'^(?:module\s*\d+|unit\s*\d+|chapter\s*\d+|\d+[\.\)]|\-|\*)\s*', '', term_str, flags=re.I).strip()
        t = re.sub(r'^(?:a|an|the)\s+', '', t, flags=re.I).strip()
        if t:
            t = t[0].upper() + t[1:]
        return t

    def clean_desc(term_str, text_str):
        if not text_str:
            return ""
        # Remove leading term repetition (with optional article e.g. "A process is ...")
        text_str = re.sub(r'^(?:(?:a|an|the)\s+)?(?:' + re.escape(term_str) + r')\s+(?:is\s+(?:a\s+|an\s+|the\s+)?|means\s+(?:that\s+)?|refers\s+to\s+)', '', text_str, flags=re.I).strip()
        # Remove personal or conversational examples
        text_str = re.sub(r'^(?:My name is\s+[A-Za-z]+,?\s*(?:and)?\s*|I am\s+(?:an?\s+)?)', '', text_str, flags=re.I).strip()
        text_str = re.split(r'\b(For example|Let’s take|Let\'s take|Considering HumanBeing|e\.g\.)\b', text_str, flags=re.I)[0].strip()
        text_str = re.sub(r'[\.\s,;:\-]+$', '', text_str).strip()
        if text_str:
            text_str = text_str[0].upper() + text_str[1:]
        return text_str

    def is_valid_term(term_str):
        t_clean = term_str.strip().lower()
        if len(t_clean) < 2 or len(t_clean) > 35:
            return False
        if t_clean.startswith(('module', 'unit', 'chapter', 'note', 'fig', 'table', 'my name', 'let', 'the different', 'which', 'example', 'overview', 'summary', 'syllabus', 'objective', 'outcome', 'course')):
            return False
        return True

    # 1. Gather all concepts & definitions from ai_analysis and extracted_pages
    concepts = []
    seen_terms = set()

    analysis = getattr(quiz, 'ai_analysis', None)
    data = analysis.structured_json if (analysis and analysis.structured_json) else {}
    defs = data.get('definitions', [])
    for d in defs:
        t = clean_term(d.get('term', ''))
        desc = clean_desc(t, d.get('definition', '').strip())
        if is_valid_term(t) and len(desc) >= 15 and t.lower() not in seen_terms:
            seen_terms.add(t.lower())
            concepts.append({
                'term': t,
                'desc': desc,
                'page': d.get('pages', [1])[0] if d.get('pages') else 1,
                'chunk_id': d.get('chunk_id', 'CHK-01'),
                'topic': t
            })

    # Also extract directly from extracted pages (headings, colon lines, bullet points)
    for p in quiz.extracted_pages.all():
        text = p.clean_text
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            t, desc = None, None
            if line.endswith(':'):
                t = line[:-1].strip()
                desc = lines[i+1].strip() if i + 1 < len(lines) else ''
            elif ':' in line:
                parts = line.split(':', 1)
                t = parts[0].strip()
                desc = parts[1].strip()
            if t and desc:
                t = clean_term(t)
                desc = clean_desc(t, desc)
                if is_valid_term(t) and len(desc) >= 15 and t.lower() not in seen_terms:
                    seen_terms.add(t.lower())
                    concepts.append({
                        'term': t,
                        'desc': desc,
                        'page': p.page_number,
                        'chunk_id': 'CHK-01',
                        'topic': t
                    })

    # Subject-aligned backup terms if document has few concepts
    s_lower = subject_name.lower()
    if 'software' in s_lower or 'oop' in s_lower or 'design' in s_lower:
        fallback_terms = ['Inheritance', 'Encapsulation', 'Polymorphism', 'Abstraction', 'Class', 'Design Pattern', 'High Cohesion', 'Low Coupling']
        fallback_descs = [
            'Hiding internal implementation details and exposing only what is necessary',
            'Enabling a routine or interface to operate on multiple data types at runtime',
            'Allowing a class to acquire properties and methods from an existing class',
            'Defining the structure, methods, and variables for creating software objects',
            'Ensuring a module or class has a small, focused set of single responsibilities',
            'Minimizing interdependence between distinct modules to improve maintainability',
        ]
    elif 'network' in s_lower:
        fallback_terms = ['Packet Switching', 'Routing Protocol', 'TCP/IP', 'Subnet Mask', 'MAC Address', 'Flow Control']
        fallback_descs = [
            'Breaking data into packets and transmitting them across dynamic network routes',
            'Guarantees reliable, ordered delivery of byte streams between networked hosts',
            'Assigns logical identifiers to network devices within an IP subnetwork',
            'Prevents a fast sender from overwhelming a slow receiver with data packets',
        ]
    elif 'os' in s_lower or 'operating' in s_lower:
        fallback_terms = ['Process Scheduling', 'Virtual Memory', 'Deadlock Prevention', 'Context Switching', 'Semaphore', 'Mutual Exclusion']
        fallback_descs = [
            'Allocating CPU time among competing active processes based on priority',
            'Providing an idealized abstraction of available memory storage to processes',
            'Saving and restoring the state of a CPU process during execution preemption',
            'Ensuring that only one process can execute a critical section at any given time',
        ]
    else:
        fallback_terms = ['Modular Decomposition', 'Systematic Verification', 'Interface Specification', 'Data Encapsulation', 'Abstraction Layer']
        fallback_descs = [
            'Breaking complex systems into independent, manageable and reusable components',
            'Ensuring implementation adherence to formal specifications and quality benchmarks',
            'Hiding lower-level hardware or data details behind standard interfaces',
        ]

    # Helper to randomize and build clean student-friendly MCQ
    def build_mcq(q_text, correct_ans, distractors, explanation, topic, diff, bloom, q_type, cid, page, quote):
        clean_d = [d for d in distractors if d.strip().lower() != correct_ans.strip().lower()]
        pad_idx = 0
        while len(clean_d) < 3:
            cand = fallback_terms[pad_idx % len(fallback_terms)] if len(correct_ans) < 35 else fallback_descs[pad_idx % len(fallback_descs)]
            if cand.strip().lower() != correct_ans.strip().lower() and cand not in clean_d:
                clean_d.append(cand)
            pad_idx += 1

        opts = [correct_ans] + clean_d[:3]
        rng = random.Random(hash(q_text))
        rng.shuffle(opts)

        opt_letters = ['A', 'B', 'C', 'D']
        correct_idx = opts.index(correct_ans)
        correct_letter = opt_letters[correct_idx]

        clean_expl = f"Correct Answer: Option {correct_letter}. {explanation}"

        return {
            'text': q_text,
            'option_a': opts[0],
            'option_b': opts[1],
            'option_c': opts[2],
            'option_d': opts[3],
            'correct_option': correct_letter,
            'explanation': clean_expl,
            'topic': topic,
            'difficulty': diff,
            'blooms_level': bloom,
            'question_type': q_type,
            'chunk_id': cid,
            'source_page': page,
            'source_quote': quote[:200],
        }

    pool_easy = []
    pool_medium = []
    pool_hard = []

    # 1. Easy questions: Identify the term from its definition
    all_terms = [c['term'] for c in concepts]
    for c in concepts:
        term = c['term']
        desc = c['desc']
        other_terms = [t for t in all_terms if t != term]
        if not other_terms:
            other_terms = [t for t in fallback_terms if t != term]

        q_text = f"Which concept in {subject_name} is described as: \"{desc}\"?"
        pool_easy.append(build_mcq(
            q_text=q_text,
            correct_ans=term,
            distractors=other_terms,
            explanation=f"'{term}' is defined as: {desc}",
            topic=term,
            diff='easy',
            bloom='Remember',
            q_type='conceptual',
            cid=c['chunk_id'],
            page=c['page'],
            quote=desc
        ))

    # 2. Medium questions: What is the primary definition or role of a term
    all_descs = [c['desc'] for c in concepts]
    for c in concepts:
        term = c['term']
        desc = c['desc']
        other_descs = [d for d in all_descs if d != desc]
        if not other_descs:
            other_descs = [d for d in fallback_descs if d != desc]

        q_text = f"What is the primary role or definition of '{term}' in {subject_name}?"
        pool_medium.append(build_mcq(
            q_text=q_text,
            correct_ans=desc,
            distractors=other_descs,
            explanation=f"In {subject_name}, '{term}' refers to: {desc}",
            topic=term,
            diff='medium',
            bloom='Understand',
            q_type='understanding',
            cid=c['chunk_id'],
            page=c['page'],
            quote=desc
        ))

    # 3. Hard / Apply questions: Advantages, Best Practices, and Core Principles
    adv_sentences = []
    for p in quiz.extracted_pages.all():
        for sent in re.split(r'[.\n]+', p.clean_text):
            sent = sent.strip()
            if any(k in sent.lower() for k in ['advantage', 'improving', 'benefit', 'characteristic', 'cohesion', 'coupling', 'reusability']):
                if 25 <= len(sent) <= 160 and not sent.lower().startswith(('module', 'unit', 'note', 'the different')):
                    adv_sentences.append((sent, p.page_number))

    for sent, page_num in adv_sentences:
        q_text = f"Which of the following is an essential advantage or principle in {subject_name}?"
        clean_ans = sent
        distractors = [
            "Eliminating the need for software testing and verification",
            "Merging all classes into a single unstructured monolithic script",
            "Restricting system execution to single-threaded sequential scripts",
            "Disabling object-oriented modularity and data hiding",
        ]
        pool_hard.append(build_mcq(
            q_text=q_text,
            correct_ans=clean_ans,
            distractors=distractors,
            explanation=f"In {subject_name}, {clean_ans}.",
            topic=subject_name,
            diff='hard',
            bloom='Apply',
            q_type='application',
            cid='CHK-01',
            page=page_num,
            quote=sent
        ))

    # Quota allocation
    diff_quotas = calculate_difficulty_quotas(target_count, quiz.difficulty)
    selected = []

    # Pick Easy
    for _ in range(diff_quotas.get('easy', 0)):
        if pool_easy:
            selected.append(pool_easy.pop(0))

    # Pick Medium
    for _ in range(diff_quotas.get('medium', 0)):
        if pool_medium:
            selected.append(pool_medium.pop(0))

    # Pick Hard
    for _ in range(diff_quotas.get('hard', 0)):
        if pool_hard:
            selected.append(pool_hard.pop(0))
        elif pool_medium:
            selected.append(pool_medium.pop(0))

    # Fill remaining from any available candidate pool
    remaining = pool_easy + pool_medium + pool_hard
    while len(selected) < target_count and remaining:
        cand = remaining.pop(0)
        if not any(s['text'] == cand['text'] for s in selected):
            selected.append(cand)

    return selected[:target_count]


def generate_quiz_mcqs(quiz: Quiz, force: bool = False) -> dict:
    """
    Main entrypoint for Phase 6, 7 & 8:
    Retrieves RAG context -> Synthesizes source-grounded MCQs under Bloom's Taxonomy & Difficulty Control -> Persists to Question model.
    """
    existing_qs = quiz.questions.all()
    if existing_qs.exists() and not force:
        return {
            'success': True,
            'already_generated': True,
            'total_questions': existing_qs.count(),
            'model_used': 'Existing Question Bank',
        }

    # Ensure prior phases have completed
    if not quiz.chunks.exists():
        return {
            'success': False,
            'error': 'Phase 3 Semantic Chunks required before generating questions.'
        }

    # Ensure Phase 5 Concept Ranking has run
    if not quiz.ranked_concepts.exists():
        rank_quiz_concepts(quiz)

    target_count = quiz.num_questions or 5
    rag_context = format_rag_context(quiz)
    api_key = get_api_key()

    model_used = "Noteforge Source-Grounded RAG (Deterministic Engine)"
    raw_questions = []

    if api_key:
        try:
            raw_questions, model_used = generate_mcqs_with_gemini(rag_context, api_key, target_count)
        except Exception as e:
            logger.warning(f"LLM MCQ generation failed, falling back to deterministic RAG generator: {e}")
            raw_questions = fallback_deterministic_mcqs(quiz, target_count)
            model_used += " [Fallback after LLM error]"
    else:
        raw_questions = fallback_deterministic_mcqs(quiz, target_count)

    # Persist to database with canonical Bloom's and Difficulty normalization
    question_records = []
    for item in raw_questions:
        text = clean_meta_question_text(item.get('text', '').strip())
        opt_a = item.get('option_a', '').strip()
        opt_b = item.get('option_b', '').strip()
        opt_c = item.get('option_c', '').strip()
        opt_d = item.get('option_d', '').strip()
        corr = item.get('correct_option', 'A').strip().upper()

        if not text or not opt_a or not opt_b:
            continue

        if corr not in ['A', 'B', 'C', 'D']:
            corr = 'A'

        expl = clean_meta_question_text(item.get('explanation', 'Ground-truth explanation from course curriculum.'))
        topic = item.get('topic', quiz.title or 'Curriculum')

        # Normalize Bloom's level strictly to canonical 4 levels
        raw_bloom = str(item.get('blooms_level', 'Understand')).strip().lower()
        if raw_bloom in ['remember', 'recall', 'knowledge', 'fact', 'memory']:
            bloom = 'Remember'
        elif raw_bloom in ['understand', 'comprehension', 'interpret', 'interpretation']:
            bloom = 'Understand'
        elif raw_bloom in ['apply', 'application', 'problem-solving', 'procedure']:
            bloom = 'Apply'
        elif raw_bloom in ['analyse', 'analyze', 'analysis', 'analytical', 'evaluate', 'evaluation']:
            bloom = 'Analyse'
        else:
            bloom = 'Understand'

        # Normalize difficulty strictly according to Phase 8 criteria:
        # Easy -> direct understanding
        # Medium -> concept connection / application
        # Hard -> reasoning / analysis
        raw_diff = str(item.get('difficulty', '')).strip().lower()
        if raw_diff in ['easy', 'medium', 'hard']:
            diff = raw_diff
        elif bloom == 'Remember':
            diff = 'easy'
        elif bloom in ['Understand', 'Apply']:
            diff = 'medium'
        elif bloom == 'Analyse':
            diff = 'hard'
        else:
            diff = 'medium'

        q_type = item.get('question_type', 'conceptual')
        cid = item.get('chunk_id', 'CHK-01')
        sp = item.get('source_page', 1)
        quote = item.get('source_quote', '')

        question_records.append(
            Question(
                quiz=quiz,
                text=text,
                marks=1,
                option_a=opt_a,
                option_b=opt_b,
                option_c=opt_c,
                option_d=opt_d,
                correct_option=corr,
                explanation=expl,
                topic=topic[:200],
                difficulty=diff[:20],
                blooms_level=bloom,
                question_type=q_type[:50],
                chunk_id=cid[:30],
                source_page=sp if isinstance(sp, int) else 1,
                source_quote=quote,
            )
        )

    # Clean previous and bulk create
    Question.objects.filter(quiz=quiz).delete()
    Question.objects.bulk_create(question_records)

    return {
        'success': True,
        'already_generated': False,
        'model_used': model_used,
        'total_questions': len(question_records),
    }
