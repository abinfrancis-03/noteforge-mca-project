from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.db.models import Avg, Max, Count, Q
from django.http import JsonResponse, HttpResponse
from django.utils import timezone

import random
import os
import re
from typing import Optional, Tuple, List, Dict, Any

from django.core.exceptions import PermissionDenied

from accounts.models import (
    User, Department, Semester, Subject, StudyMaterial, MaterialDownload,
    Quiz, Question, QuestionOption, QuizAttempt, StudentAnswer, AuditLog, SavedSummary
)
from services.document_extractor import (
    extract_document,
    UnsupportedFileFormatException,
    EmptyDocumentException,
    DocumentExtractionException,
)
from services.ai_summarizer import AISummarizerService
from services.mindmap_image_generator import extract_infographic_data, render_mindmap_image
from .topic_analytics import run_topic_performance_analytics, get_student_cumulative_topic_analytics
from .personalized_practice import run_personalized_practice_flow

MOTIVATIONAL_QUOTES = [
    "The future belongs to those who believe in the beauty of their dreams.",
    "Believe you can, and you’re halfway there.",
    "Start where you are, use what you have, do what you can.",
    "Every expert was once a beginner.",
    "Difficult roads often lead to beautiful destinations.",
    "A journey of a thousand miles begins with a single step.",
    "Shoot for the moon. Even if you miss, you’ll land among the stars.",
    "Your limitation—it’s only your imagination.",
    "Dream it. Believe it. Achieve it.",
    "Great things never come from comfort zones.",
]



def student_required(view_func):
    """Decorator: login + student role required + first login password check."""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        if request.user.role != 'student':
            messages.error(request, 'Access denied. Student portal only.')
            return redirect('login')
        if request.user.is_first_login and view_func.__name__ != 'change_password':
            messages.warning(request, 'Please change your temporary password before proceeding.')
            return redirect('student_change_password')
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


@student_required
def dashboard(request):
    student = request.user
    department = student.department
    semester = student.semester
    batch = student.batch

    # Subjects belonging to student's semester & department
    subjects = Subject.objects.filter(semester=semester)
    if department:
        subjects = subjects.filter(semester__department=department)

    # Materials for these subjects
    materials = StudyMaterial.objects.filter(subject__in=subjects).select_related('subject', 'uploaded_by').order_by('-created_at')

    # Published Quizzes for these subjects & batch
    quizzes_qs = Quiz.objects.filter(
        subject__in=subjects,
        is_published=True,
        is_practice=False
    ).select_related('subject', 'created_by').order_by('-published_at', '-created_at')

    if batch:
        quizzes_qs = quizzes_qs.filter(Q(batch=batch) | Q(batch__isnull=True))

    attempts = QuizAttempt.objects.filter(student=student).select_related('quiz', 'quiz__subject')
    
    # Calculate stats
    completed_quiz_ids = set(attempts.values_list('quiz_id', flat=True))
    avg_score_raw = attempts.aggregate(Avg('percentage'))['percentage__avg']
    avg_score = round(avg_score_raw, 1) if avg_score_raw is not None else 0.0

    # ── 1. Academic Grade Band ──
    if attempts.exists():
        if avg_score >= 85:
            grade_band = 'Excellent'
            grade_band_color = '#10b981'
            grade_badge_bg = 'bg-success-subtle text-success border border-success-subtle'
            grade_desc = 'Demonstrating excellent conceptual understanding and high scores.'
        elif avg_score >= 70:
            grade_band = 'Good'
            grade_band_color = '#4f46e5'
            grade_badge_bg = 'bg-primary-subtle text-primary border border-primary-subtle'
            grade_desc = 'Solid grasp of core curriculum concepts with steady performance.'
        elif avg_score >= 50:
            grade_band = 'Average'
            grade_band_color = '#f59e0b'
            grade_badge_bg = 'bg-warning-subtle text-warning-emphasis border border-warning-subtle'
            grade_desc = 'Passing baseline achieved. Regular revision will boost your scores.'
        else:
            grade_band = 'Needs Improvement'
            grade_band_color = '#ef4444'
            grade_badge_bg = 'bg-danger-subtle text-danger border border-danger-subtle'
            grade_desc = 'Recommended to review study notes and attempt targeted practice.'
    else:
        grade_band = 'Not Evaluated'
        grade_band_color = '#94a3b8'
        grade_badge_bg = 'bg-light text-secondary border'
        grade_desc = 'Complete your first quiz assessment to generate your academic grade band.'

    # ── 2. Curriculum Coverage ──
    total_materials_count = materials.count()
    downloaded_material_ids = set(MaterialDownload.objects.filter(student=student, material__in=materials).values_list('material_id', flat=True))
    saved_summary_ids = set(SavedSummary.objects.filter(student=student, source_material__in=materials).values_list('source_material_id', flat=True))
    
    explored_material_ids = downloaded_material_ids | saved_summary_ids
    materials_read_count = min(len(explored_material_ids), total_materials_count)
    materials_remaining_count = max(0, total_materials_count - materials_read_count)
    curriculum_coverage_pct = round((materials_read_count / total_materials_count * 100), 1) if total_materials_count > 0 else 0.0

    # ── 3. Upcoming & Pending Quizzes ──
    now_time = timezone.now()
    pending_quizzes = []
    for q in quizzes_qs:
        if q.pk not in completed_quiz_ids:
            ref_time = q.published_at or q.created_at
            q.is_new = (now_time - ref_time).total_seconds() < (86400 * 5) if ref_time else False
            pending_quizzes.append(q)

    # Check for completed / archived previous semesters
    previous_semesters_count = 0
    if department and semester and semester.number > 1:
        previous_semesters_count = Semester.objects.filter(
            department=department,
            number__lt=semester.number
        ).count()

    daily_quote = random.choice(MOTIVATIONAL_QUOTES)

    return render(request, 'student_panel/dashboard.html', {
        'student': student,
        'department': department,
        'semester': semester,
        'batch': batch,
        'subjects_count': subjects.count(),
        'materials_count': total_materials_count,
        'completed_quizzes_count': len(completed_quiz_ids),
        'avg_score': avg_score,
        'grade_band': grade_band,
        'grade_band_color': grade_band_color,
        'grade_badge_bg': grade_badge_bg,
        'grade_desc': grade_desc,
        'total_materials_count': total_materials_count,
        'materials_read_count': materials_read_count,
        'materials_remaining_count': materials_remaining_count,
        'curriculum_coverage_pct': curriculum_coverage_pct,
        'pending_quizzes': pending_quizzes,
        'previous_semesters_count': previous_semesters_count,
        'daily_quote': daily_quote,
    })



@student_required
def material_list(request):
    student = request.user
    department = student.department
    current_sem = student.semester

    # All semesters up to the student's current semester (including archived ones)
    available_semesters = []
    if department and current_sem:
        available_semesters = list(Semester.objects.filter(
            department=department,
            number__lte=current_sem.number
        ).order_by('number'))

    # Selected semester to view (default to active current semester)
    sem_id = request.GET.get('sem')
    viewing_sem = current_sem
    if sem_id and available_semesters:
        for s in available_semesters:
            if str(s.pk) == str(sem_id):
                viewing_sem = s
                break

    is_archived = bool(viewing_sem and current_sem and viewing_sem.number < current_sem.number)

    subjects = Subject.objects.filter(semester=viewing_sem)
    if department:
        subjects = subjects.filter(semester__department=department)

    sub_id = request.GET.get('sub')
    materials = StudyMaterial.objects.filter(subject__in=subjects).select_related('subject', 'uploaded_by')

    selected_sub = None
    if sub_id:
        selected_sub = get_object_or_404(Subject, pk=sub_id, pk__in=subjects.values_list('id', flat=True))
        materials = materials.filter(subject=selected_sub)

    return render(request, 'student_panel/material_list.html', {
        'student': student,
        'materials': materials.order_by('-created_at'),
        'subjects': subjects,
        'selected_sub': selected_sub,
        'available_semesters': available_semesters,
        'viewing_sem': viewing_sem,
        'current_sem': current_sem,
        'is_archived': is_archived,
    })


@student_required
def download_material(request, pk):
    """Tracks material download for curriculum coverage and redirects to the file."""
    material = get_object_or_404(StudyMaterial, pk=pk)

    # Record download for curriculum coverage tracking
    MaterialDownload.objects.get_or_create(student=request.user, material=material)

    if material.file:
        return redirect(material.file.url)
    elif material.external_link:
        return redirect(material.external_link)
    else:
        messages.error(request, 'No downloadable file is attached to this study material.')
        return redirect('student_material_list')


@student_required
def quiz_list(request):
    student = request.user
    subjects = Subject.objects.filter(semester=student.semester)
    if student.department:
        subjects = subjects.filter(semester__department=student.department)

    quizzes = Quiz.objects.filter(subject__in=subjects, is_published=True, is_practice=False).select_related('subject')
    if student.batch:
        quizzes = quizzes.filter(Q(batch=student.batch) | Q(batch__isnull=True))

    sub_id = request.GET.get('sub')
    selected_sub = None
    if sub_id:
        selected_sub = get_object_or_404(Subject, pk=sub_id, pk__in=subjects.values_list('id', flat=True))
        quizzes = quizzes.filter(subject=selected_sub)

    # Phase 15: Personalized practice quizzes for this student
    practice_quizzes_qs = Quiz.objects.filter(
        practice_student=student,
        is_practice=True,
        is_published=True
    ).select_related('subject', 'parent_quiz').order_by('-created_at')
    if selected_sub:
        practice_quizzes_qs = practice_quizzes_qs.filter(subject=selected_sub)

    attempts = QuizAttempt.objects.filter(student=student)
    attempts_map = {}
    for att in attempts:
        qid = att.quiz_id
        if qid not in attempts_map:
            attempts_map[qid] = []
        attempts_map[qid].append(att)

    # Attach attempt summaries to quiz items
    quiz_data = []
    for q in quizzes:
        att_list = attempts_map.get(q.pk, [])
        best_att = max(att_list, key=lambda a: a.percentage) if att_list else None
        quiz_data.append({
            'quiz': q,
            'attempt_count': len(att_list),
            'best_attempt': best_att,
            'latest_attempt': att_list[0] if att_list else None
        })

    # Attach attempt summaries to practice quizzes
    practice_data = []
    for pq in practice_quizzes_qs:
        p_att_list = attempts_map.get(pq.pk, [])
        p_best_att = max(p_att_list, key=lambda a: a.percentage) if p_att_list else None
        practice_data.append({
            'quiz': pq,
            'attempt_count': len(p_att_list),
            'best_attempt': p_best_att,
            'latest_attempt': p_att_list[0] if p_att_list else None
        })

    return render(request, 'student_panel/quiz_list.html', {
        'quiz_data': quiz_data,
        'practice_data': practice_data,
        'subjects': subjects,
        'selected_sub': selected_sub,
    })


@student_required
def generate_personalized_practice(request, pk):
    """
    Phase 15: Adaptive RAG-based personalized practice test generator.
    Identifies the student's weak topic, retrieves relevant PDF chunks,
    synthesizes fresh non-duplicate MCQs (Understand & Apply), validates them,
    provisions a personalized practice quiz, and redirects student to take it.
    """
    student = request.user
    subjects = Subject.objects.filter(semester=student.semester)
    if student.department:
        subjects = subjects.filter(semester__department=student.department)

    source_quiz = get_object_or_404(Quiz, pk=pk, subject__in=subjects)
    weak_topic = (request.GET.get('topic') or request.POST.get('topic') or '').strip()

    if not weak_topic:
        # If no topic explicitly passed, pick the lowest-scoring topic from latest attempt
        latest_attempt = QuizAttempt.objects.filter(student=student, quiz=source_quiz).order_by('-completed_at').first()
        if latest_attempt and latest_attempt.topic_analysis:
            weak_topics = latest_attempt.topic_analysis.get('weak_topics', [])
            if weak_topics:
                weak_topic = weak_topics[0]

    if not weak_topic:
        weak_topic = source_quiz.title

    try:
        practice_quiz = run_personalized_practice_flow(
            student=student,
            source_quiz=source_quiz,
            weak_topic=weak_topic,
            target_count=4,
            use_ai=True
        )
        messages.success(
            request,
            f"🎯 Generated adaptive personalized practice for '{weak_topic}'! Target: Understanding & Application."
        )
        return redirect('student_quiz_take', pk=practice_quiz.pk)
    except Exception as e:
        messages.error(request, f"Could not generate practice test: {str(e)}")
        return redirect('student_quiz_list')


@student_required
def quiz_take(request, pk):
    student = request.user
    subjects = Subject.objects.filter(semester=student.semester)
    if student.department:
        subjects = subjects.filter(semester__department=student.department)

    quiz = get_object_or_404(Quiz, pk=pk, is_published=True)
    # Phase 15: Access control for personalized practice quizzes vs regular quizzes
    if quiz.is_practice:
        if quiz.practice_student != student:
            messages.error(request, 'Access denied to this personalized practice test.')
            return redirect('student_quiz_list')
    else:
        if quiz.subject not in subjects:
            messages.error(request, 'This quiz does not belong to your enrolled curriculum.')
            return redirect('student_quiz_list')
        if student.batch and quiz.batch and quiz.batch != student.batch:
            messages.error(request, 'This quiz is not assigned to your batch.')
            return redirect('student_quiz_list')

    # Phase 12: A question must not become visible to students until the teacher approves and publishes the quiz
    questions = list(
        quiz.questions.filter(
            teacher_approved=True,
            is_approved=True,
            approval_status='approved'
        ).prefetch_related('options').order_by('id')
    )
    if not questions:
        messages.warning(request, 'This quiz currently has no approved questions.')
        return redirect('student_quiz_list')

    # Phase 13: Attempt number calculation for student test taking
    previous_attempts_count = QuizAttempt.objects.filter(student=student, quiz=quiz).count()
    attempt_number = previous_attempts_count + 1

    return render(request, 'student_panel/quiz_take.html', {
        'quiz': quiz,
        'questions': questions,
        'attempt_number': attempt_number,
        'time_limit_secs': quiz.time_limit_mins * 60,
    })


@student_required
def quiz_submit(request, pk):
    """
    Phase 13: Conventional Online Assessment Submission & Scoring Engine.
    Evaluates student answers, calculates score and percentage, records elapsed time,
    and persists both QuizAttempt (with attempt_number) and relational StudentAnswer records.
    """
    if request.method != 'POST':
        return redirect('student_quiz_list')

    student = request.user
    subjects = Subject.objects.filter(semester=student.semester)
    if student.department:
        subjects = subjects.filter(semester__department=student.department)

    quiz = get_object_or_404(Quiz, pk=pk, is_published=True)
    if quiz.is_practice:
        if quiz.practice_student != student:
            messages.error(request, 'Access denied to this personalized practice test.')
            return redirect('student_quiz_list')
    else:
        if quiz.subject not in subjects:
            messages.error(request, 'This quiz does not belong to your enrolled curriculum.')
            return redirect('student_quiz_list')
    # Only approved questions are graded
    questions = quiz.questions.filter(
        teacher_approved=True,
        is_approved=True,
        approval_status='approved'
    ).prefetch_related('options').order_by('id')

    try:
        time_taken_secs = int(request.POST.get('time_taken_seconds', 0))
    except ValueError:
        time_taken_secs = 0

    score = 0
    total_questions = questions.count()
    answers_details = {}
    student_answers_batch = []

    for q in questions:
        chosen_opt = request.POST.get(f'q_{q.pk}', '').strip().upper()
        is_correct = (chosen_opt == q.correct_option.upper())
        if is_correct:
            score += 1

        # Determine option text
        mapping = {'A': q.option_a, 'B': q.option_b, 'C': q.option_c, 'D': q.option_d}
        chosen_text = mapping.get(chosen_opt, '')

        answers_details[str(q.pk)] = {
            'question_text': q.text,
            'option_a': q.option_a,
            'option_b': q.option_b,
            'option_c': q.option_c,
            'option_d': q.option_d,
            'chosen_option': chosen_opt,
            'chosen_answer': chosen_text,
            'correct_option': q.correct_option,
            'is_correct': is_correct,
            'explanation': q.explanation,
            'topic': q.topic,
            'difficulty': q.difficulty,
            'blooms_level': q.blooms_level,
        }

        student_answers_batch.append({
            'question': q,
            'selected_option': chosen_opt,
            'selected_answer': chosen_text,
            'is_correct': is_correct,
            'marks_obtained': 1 if is_correct else 0,
        })

    percentage = round((score / total_questions * 100), 1) if total_questions > 0 else 0.0

    # Phase 13: Sequential attempt tracking
    prev_attempts = QuizAttempt.objects.filter(student=student, quiz=quiz).count()
    attempt_number = prev_attempts + 1

    attempt = QuizAttempt.objects.create(
        student=student,
        quiz=quiz,
        attempt_number=attempt_number,
        score=score,
        total_questions=total_questions,
        percentage=percentage,
        time_taken_seconds=time_taken_secs,
        answers_json=answers_details
    )

    # Persist relational StudentAnswer instances for each question
    for ans in student_answers_batch:
        StudentAnswer.objects.create(
            attempt=attempt,
            question=ans['question'],
            selected_option=ans['selected_option'],
            selected_answer=ans['selected_answer'],
            is_correct=ans['is_correct'],
            marks_obtained=ans['marks_obtained'],
        )

    # Phase 14: Compute and persist topic-wise performance analytics & recommendations
    run_topic_performance_analytics(attempt)

    AuditLog.objects.create(
        actor=student,
        action='create',
        target_model='QuizAttempt',
        description=f"{student.username} completed quiz '{quiz.title}' (Attempt #{attempt_number}) with score {score}/{total_questions} ({percentage}%)."
    )

    messages.success(request, f'Quiz submitted! Attempt #{attempt_number} recorded: Score {score}/{total_questions} ({percentage}%).')
    return redirect('student_quiz_result', pk=quiz.pk, attempt_id=attempt.pk)


@student_required
def quiz_result(request, pk, attempt_id):
    student = request.user
    quiz = get_object_or_404(Quiz, pk=pk)
    attempt = get_object_or_404(QuizAttempt, pk=attempt_id, quiz=quiz, student=student)

    # Format time taken
    mins = attempt.time_taken_seconds // 60
    secs = attempt.time_taken_seconds % 60
    time_formatted = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

    student_answers = attempt.student_answers.select_related('question').all()

    # Phase 14: Retrieve or compute topic-wise performance analytics
    topic_analysis = attempt.topic_analysis
    if not topic_analysis or 'topics' not in topic_analysis:
        topic_analysis = run_topic_performance_analytics(attempt)

    return render(request, 'student_panel/quiz_result.html', {
        'quiz': quiz,
        'attempt': attempt,
        'answers': attempt.answers_json,
        'student_answers': student_answers,
        'time_formatted': time_formatted,
        'topic_analysis': topic_analysis,
    })


@student_required
def student_performance_analytics(request):
    """
    Phase 14: Topic-wise learning analytics cumulative student portal.
    Displays cumulative topic mastery, identifies persistent weak areas,
    and lists prioritized revision study recommendations without personal or psychological judgment.
    """
    student = request.user
    department = student.department
    semester = student.semester

    subjects = Subject.objects.filter(semester=semester)
    if department:
        subjects = subjects.filter(semester__department=department)

    subject_id = request.GET.get('subject')
    selected_subject = None
    if subject_id:
        selected_subject = subjects.filter(pk=subject_id).first()

    cumulative_data = get_student_cumulative_topic_analytics(student, subject=selected_subject)
    recent_attempts = QuizAttempt.objects.filter(student=student).select_related('quiz', 'quiz__subject').order_by('-completed_at')[:8]

    return render(request, 'student_panel/analytics.html', {
        'subjects': subjects,
        'selected_subject': selected_subject,
        'analytics': cumulative_data,
        'recent_attempts': recent_attempts,
    })


@login_required
def change_password(request):
    if request.user.role != 'student':
        messages.error(request, 'Access denied.')
        return redirect('login')

    if request.method == 'POST':
        old_pw = request.POST.get('old_password', '')
        new_pw1 = request.POST.get('new_password1', '')
        new_pw2 = request.POST.get('new_password2', '')

        if not request.user.check_password(old_pw):
            messages.error(request, 'Current password is incorrect.')
        elif new_pw1 != new_pw2:
            messages.error(request, 'New passwords do not match.')
        elif len(new_pw1) < 6:
            messages.error(request, 'Password must be at least 6 characters.')
        else:
            request.user.set_password(new_pw1)
            request.user.is_first_login = False
            request.user.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, 'Password updated successfully!')
            return redirect('student_dashboard')

    return render(request, 'student_panel/change_password.html')


# ─── 1. On-Demand AI Document Summarizer ──────────────────────────────────────

@student_required
def ai_summary_hub(request):
    student = request.user
    materials = StudyMaterial.objects.filter(
        subject__semester=student.semester
    ).select_related('subject', 'uploaded_by')

    if student.department:
        materials = materials.filter(subject__semester__department=student.department)

    query = request.GET.get('q', '').strip()
    if query:
        materials = materials.filter(
            Q(title__icontains=query) | Q(subject__name__icontains=query) | Q(subject__subject_code__icontains=query)
        )

    sub_id = request.GET.get('sub', '').strip()
    if sub_id:
        materials = materials.filter(subject_id=sub_id)

    subjects = Subject.objects.filter(semester=student.semester)
    if student.department:
        subjects = subjects.filter(semester__department=student.department)

    saved_material_ids = set(SavedSummary.objects.filter(student=student, source_material__isnull=False).values_list('source_material_id', flat=True))

    return render(request, 'student_panel/ai_summary_hub.html', {
        'materials': materials,
        'subjects': subjects,
        'selected_sub': sub_id,
        'query': query,
        'saved_material_ids': saved_material_ids,
    })


def get_gemini_api_key():
    from django.conf import settings
    env_file = settings.BASE_DIR / '.env'
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass
    return (
        os.environ.get('GEMINI_API_KEY')
        or getattr(settings, 'GEMINI_API_KEY', '')
        or os.environ.get('GOOGLE_API_KEY')
        or ''
    )


def call_gemini_rest(prompt, api_key, model_names=None):
    if not api_key:
        return None, None
    if model_names is None:
        model_names = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash', 'gemini-flash-latest', 'gemini-2.5-pro']
    import requests
    headers = {'Content-Type': 'application/json'}
    for m in model_names:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}'
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.2,
                'maxOutputTokens': 8192,
            }
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    parts = candidates[0].get('content', {}).get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text'].strip(), f'Google Gemini ({m})'
        except Exception:
            continue
    return None, None


@student_required
def save_gemini_key(request):
    if request.method == 'POST':
        key = request.POST.get('gemini_api_key', '').strip()
        from django.conf import settings
        env_file = settings.BASE_DIR / '.env'
        existing_lines = []
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    existing_lines = f.readlines()
            except Exception:
                existing_lines = []
        
        updated = False
        new_lines = []
        for line in existing_lines:
            if line.strip().startswith('GEMINI_API_KEY='):
                new_lines.append(f'GEMINI_API_KEY={key}\n')
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f'GEMINI_API_KEY={key}\n')
            
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            os.environ['GEMINI_API_KEY'] = key
            messages.success(request, 'Google Gemini API key saved! Live Gemini AI generation is now active.')
        except Exception as e:
            messages.error(request, f'Could not save API key to .env: {e}')

    next_url = request.POST.get('next') or request.GET.get('next') or 'student_ai_summary_hub'
    return redirect(next_url)


def check_student_material_access(student, material) -> bool:
    """
    Security check: verify logged-in student has legitimate access to this study material.
    Denies access if material belongs to a subject outside student's semester or department.
    """
    if not student.is_authenticated or getattr(student, 'role', '') != 'student':
        return False
    if student.semester_id and material.subject.semester_id != student.semester_id:
        return False
    if student.department_id:
        material_dept_id = material.subject.semester.department_id
        if material_dept_id and material_dept_id != student.department_id:
            return False
    return True


def extract_text_from_material(material) -> str:
    """
    Format-independent text extractor using the document extractor factory.
    """
    if material.file:
        try:
            doc = extract_document(material.file, document_title=material.title)
            text = doc.get_combined_text()
            if text.strip():
                return text.strip()
        except Exception:
            pass

    desc = material.description or ""
    return f"{material.title}\n{desc}".strip()


@student_required
def document_summarize(request, pk):
    """
    AI Study Summary Engine:
    - Material-specific & format-independent (PDF, PPTX, DOCX, TXT)
    - Verifies student ownership/access security
    - Checks for existing cached summary before calling AI
    - Two-level summarization with structured JSON and exam-ready markdown
    """
    student = request.user
    material = get_object_or_404(
        StudyMaterial.objects.select_related('subject', 'subject__semester', 'subject__semester__department'),
        pk=pk
    )

    # 1. Security Check: Never trust material ID blindly from browser
    if not check_student_material_access(student, material):
        raise PermissionDenied("Access Denied: You do not have permission to view or summarize this study material.")

    # 2. Existing Summary Check:
    is_regenerate = request.GET.get('regenerate') == '1' or (request.method == 'POST' and request.POST.get('action') == 'regenerate')

    summary_text = ""
    engine_name = "AI Study Engine"
    extracted_length = 0
    error_message = None
    api_key = get_gemini_api_key()

    if not is_regenerate and material.ai_summary:
        # Existing summary present -> Return immediately, do NOT call AI API
        summary_text = material.ai_summary
        engine_name = "Cached AI Study Summary"
        extracted_length = len(summary_text)
    else:
        # Need to generate summary
        if not material.file:
            error_message = "No file is attached to this study material for AI summarization."
        else:
            try:
                # Format-independent extraction
                normalized_doc = extract_document(material.file, document_title=material.title)
                extracted_length = len(normalized_doc.get_combined_text())

                if not api_key:
                    engine_name = "Gemini API Key Required"
                    error_message = "Google Gemini API key is required. Please click 'Connect Gemini API' to configure your key."
                else:
                    summarizer = AISummarizerService(api_key=api_key)
                    summary_text, structured_json, engine_name = summarizer.summarize_document(normalized_doc)

                    # Cache directly to material
                    material.ai_summary = summary_text
                    material.ai_summary_json = structured_json
                    material.ai_summary_updated_at = timezone.now()
                    material.save(update_fields=['ai_summary', 'ai_summary_json', 'ai_summary_updated_at'])

            except UnsupportedFileFormatException as e:
                error_message = str(e)
            except EmptyDocumentException as e:
                error_message = str(e)
            except DocumentExtractionException as e:
                error_message = f"Document extraction failed: {str(e)}"
            except Exception as e:
                error_message = f"AI Summarization could not be completed: {str(e)}"

    # 3. Handle Save to Library
    if request.method == 'POST' and request.POST.get('action') == 'save_summary':
        to_save = summary_text or material.ai_summary
        if to_save:
            SavedSummary.objects.create(
                student=student,
                source_material=material,
                title=f"Summary: {material.title}",
                summary_text=to_save,
                summary_type='document'
            )
            messages.success(request, f'Summary for "{material.title}" saved to your Library!')
            return redirect('student_my_library')
        else:
            messages.error(request, 'No summary available to save.')

    return render(request, 'student_panel/document_summary.html', {
        'material': material,
        'summary_text': summary_text,
        'extracted_length': extracted_length,
        'engine_name': engine_name,
        'has_gemini_key': bool(api_key),
        'error_message': error_message,
        'is_cached': bool(not is_regenerate and material.ai_summary),
        'updated_at': material.ai_summary_updated_at,
    })


# ─── 2. My Library Workspace ──────────────────────────────────────────────────

@student_required
def my_library(request):
    student = request.user
    summaries = SavedSummary.objects.filter(student=student)
    return render(request, 'student_panel/my_library.html', {
        'summaries': summaries,
    })


@student_required
def delete_saved_summary(request, pk):
    student = request.user
    summary = get_object_or_404(SavedSummary, pk=pk, student=student)
    title = summary.title
    summary.delete()
    messages.success(request, f'Removed "{title}" from your Library.')
    return redirect('student_my_library')


# ─── 3. Video-to-Text Lecture Summarizer ──────────────────────────────────────

def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from various URL formats."""
    if not url:
        return None
    patterns = [
        r'(?:v=|\/embed\/|\/watch\?v=|\/v\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})',
        r'([0-9A-Za-z_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_youtube_video_title(url: str, video_id: str) -> str:
    """Fetch real video title using YouTube oEmbed API without requiring an API key."""
    import urllib.request
    import json
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(oembed_url, headers={'User-Agent': 'Mozilla/5.0 NoteForge/1.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            title = data.get('title', '').strip()
            if title:
                return title
    except Exception:
        pass
    return f"YouTube Lecture ({video_id})"


def fetch_youtube_transcript(video_id: str) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Retrieves full transcript from YouTube video supporting youtube-transcript-api 1.2+.
    Returns: (combined_text, snippets_list, error_message)
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
    except Exception as e:
        return "", [], f"YouTube transcript engine is unavailable: {str(e)}"

    snippets = []
    # 1. Try finding available transcripts through list()
    try:
        t_list = api.list(video_id)
        transcript_obj = None
        try:
            transcript_obj = t_list.find_transcript(['en', 'en-US', 'en-GB', 'en-CA', 'en-IN'])
        except Exception:
            # Fallback to any transcript available, translating to English if translatable
            first_t = next(iter(t_list), None)
            if first_t:
                if getattr(first_t, 'is_translatable', False):
                    try:
                        transcript_obj = first_t.translate('en')
                    except Exception:
                        transcript_obj = first_t
                else:
                    transcript_obj = first_t

        if transcript_obj:
            fetched = transcript_obj.fetch()
            snippets = [{'text': s.text, 'start': getattr(s, 'start', 0), 'duration': getattr(s, 'duration', 0)} for s in fetched.snippets]
    except Exception:
        pass

    # 2. If list() didn't work, try direct fetch()
    if not snippets:
        try:
            fetched = api.fetch(video_id)
            snippets = [{'text': s.text, 'start': getattr(s, 'start', 0), 'duration': getattr(s, 'duration', 0)} for s in fetched.snippets]
        except Exception:
            pass

    if not snippets:
        return "", [], "Could not retrieve a transcript for this video. Please ensure the YouTube video has closed captions/subtitles (CC) enabled."

    full_text = " ".join([s['text'] for s in snippets if s.get('text')]).strip()
    return full_text, snippets, ""


@student_required
def youtube_summarizer(request):
    student = request.user
    youtube_url = request.GET.get('url', '').strip() or request.POST.get('url', '').strip()
    action = request.POST.get('action', '').strip() if request.method == 'POST' else ''

    video_id = None
    video_title = ""
    transcript_text = ""
    summary_notes = ""
    structured_json = {}
    engine_name = "AI Video Study Engine"
    extracted_length = 0
    error_message = None
    api_key = get_gemini_api_key()

    # 1. Handle Instant Save to Library POST
    if request.method == 'POST' and action == 'save_summary':
        cached_notes = request.POST.get('summary_text', '').strip()
        if cached_notes:
            video_id = extract_youtube_video_id(youtube_url)
            saved_title = fetch_youtube_video_title(youtube_url, video_id) if video_id else "YouTube Lecture"
            SavedSummary.objects.create(
                student=student,
                title=saved_title[:200],
                summary_text=cached_notes,
                summary_type='youtube'
            )
            messages.success(request, f'Saved "{saved_title}" to your Library!')
            return redirect('student_my_library')

    # 2. Handle Instant PDF Download POST
    if request.method == 'POST' and action == 'download_pdf':
        cached_notes = request.POST.get('summary_text', '').strip()
        if cached_notes:
            video_id = extract_youtube_video_id(youtube_url)
            display_title = fetch_youtube_video_title(youtube_url, video_id) if video_id else "YouTube Lecture"
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            import io

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Title'],
                fontSize=18,
                leading=22,
                textColor=colors.HexColor('#1e293b'),
                alignment=0,
                spaceAfter=12
            )
            h2_style = ParagraphStyle(
                'H2',
                parent=styles['Heading2'],
                fontSize=14,
                leading=18,
                textColor=colors.HexColor('#2563eb'),
                spaceBefore=14,
                spaceAfter=6
            )
            h3_style = ParagraphStyle(
                'H3',
                parent=styles['Heading3'],
                fontSize=11,
                leading=15,
                textColor=colors.HexColor('#0f172a'),
                spaceBefore=8,
                spaceAfter=4
            )
            body_style = ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=9.5,
                leading=14,
                textColor=colors.HexColor('#334155'),
                spaceAfter=4
            )

            story = []
            story.append(Paragraph(f"<b>Noteforge — {display_title}</b>", title_style))
            story.append(Paragraph(f"<b>Source URL:</b> <font color='#2563eb'>{youtube_url}</font>", body_style))
            story.append(Spacer(1, 10))

            for line in cached_notes.split('\n'):
                line_str = line.strip()
                if not line_str:
                    story.append(Spacer(1, 4))
                    continue
                if line_str.startswith('# '):
                    story.append(Paragraph(f"<b>{line_str[2:]}</b>", title_style))
                elif line_str.startswith('## '):
                    story.append(Paragraph(f"<b>{line_str[3:]}</b>", h2_style))
                elif line_str.startswith('### '):
                    story.append(Paragraph(f"<b>{line_str[4:]}</b>", h3_style))
                elif line_str.startswith('- ') or line_str.startswith('* '):
                    clean_line = line_str[2:].replace('**', '<b>').replace('**', '</b>')
                    story.append(Paragraph(f"• {clean_line}", body_style))
                elif line_str.startswith('|'):
                    clean_line = " &nbsp;|&nbsp; ".join([c.strip() for c in line_str.strip('|').split('|') if c.strip()])
                    if clean_line and '---' not in clean_line:
                        story.append(Paragraph(f"<i>{clean_line}</i>", body_style))
                else:
                    clean_line = line_str.replace('**', '<b>').replace('**', '</b>')
                    story.append(Paragraph(clean_line, body_style))

            doc.build(story)
            buffer.seek(0)

            response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
            safe_filename = re.sub(r'[^A-Za-z0-9_]+', '_', display_title)[:40]
            response['Content-Disposition'] = f'attachment; filename="YouTube_Summary_{safe_filename}.pdf"'
            return response

    if youtube_url:
        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            error_message = "Invalid YouTube URL. Please provide a valid YouTube link (e.g. https://www.youtube.com/watch?v=...)."
        else:
            video_title = fetch_youtube_video_title(youtube_url, video_id)
            transcript_text, snippets, transcript_error = fetch_youtube_transcript(video_id)

            if transcript_error:
                error_message = transcript_error
            elif not transcript_text:
                error_message = "The retrieved video transcript contains no readable text."
            else:
                extracted_length = len(transcript_text)
                from services.document_extractor.base import NormalizedDocument, NormalizedSection
                from services.ai_summarizer import AISummarizerService, synthesize_local_structured_notes

                doc = NormalizedDocument(
                    document_title=video_title,
                    source_type='youtube',
                    total_units=1,
                    sections=[
                        NormalizedSection(
                            title=video_title,
                            content=transcript_text,
                            source_reference={'type': 'video', 'number': video_id}
                        )
                    ],
                    raw_text=transcript_text
                )

                try:
                    summarizer = AISummarizerService(api_key=api_key)
                    summary_notes, structured_json, engine_name = summarizer.summarize_document(doc)
                except Exception as e:
                    summary_notes, structured_json, engine_name = synthesize_local_structured_notes(doc)

    return render(request, 'student_panel/youtube_summarizer.html', {
        'youtube_url': youtube_url,
        'video_id': video_id,
        'video_title': video_title,
        'summary_notes': summary_notes,
        'structured_json': structured_json,
        'engine_name': engine_name,
        'extracted_length': extracted_length,
        'has_gemini_key': bool(api_key),
        'error_message': error_message,
    })


# ─── 4. Guardrailed Contextual Chatbot (RAG) ──────────────────────────────────

@student_required
def chatbot_hub(request):
    student = request.user
    subjects = Subject.objects.all()
    if student.department:
        subjects = subjects.filter(semester__department=student.department)
    if student.semester and hasattr(student.semester, 'number') and student.semester.number:
        subjects = subjects.filter(semester__number__lte=student.semester.number).order_by('semester__number', 'name')
    elif student.semester:
        subjects = subjects.filter(semester=student.semester)

    sub_id = request.GET.get('sub', '').strip()
    if sub_id and sub_id != 'all':
        return redirect('student_subject_chatbot', pk=sub_id)

    total_materials = StudyMaterial.objects.filter(subject__in=subjects).count()
    return render(request, 'student_panel/subject_chatbot.html', {
        'subject': None,
        'subjects': subjects,
        'materials_count': total_materials,
        'is_all_subjects': True,
    })


@student_required
def subject_chatbot(request, pk):
    student = request.user
    subjects = Subject.objects.all()
    if student.department:
        subjects = subjects.filter(semester__department=student.department)
    if student.semester and hasattr(student.semester, 'number') and student.semester.number:
        subjects = subjects.filter(semester__number__lte=student.semester.number).order_by('semester__number', 'name')
    elif student.semester:
        subjects = subjects.filter(semester=student.semester)

    if pk == 0 or str(pk).lower() == 'all':
        return redirect('student_chatbot_hub')

    subject = get_object_or_404(Subject, pk=pk, pk__in=subjects.values_list('id', flat=True))
    materials = StudyMaterial.objects.filter(subject=subject)

    return render(request, 'student_panel/subject_chatbot.html', {
        'subject': subject,
        'subjects': subjects,
        'materials_count': materials.count(),
        'is_all_subjects': False,
    })


@student_required
def api_chatbot_query(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    import json
    try:
        data = json.loads(request.body)
    except Exception:
        data = {}

    subject_id = data.get('subject_id')
    user_query = data.get('query', '').strip()
    history = data.get('history', [])

    if not user_query:
        return JsonResponse({
            'response': 'Please enter a question.',
            'sources': [],
            'status': 'empty_query',
            'is_grounded': True,
        })

    student = request.user
    try:
        from services.rag_engine import RAGEngine
        engine = RAGEngine()
        result = engine.answer_question(
            student=student,
            query=user_query,
            subject_id=subject_id,
            conversation_history=history,
        )

        return JsonResponse({
            'response': result.get('answer', ''),
            'sources': result.get('sources', []),
            'status': result.get('status', 'ok'),
            'is_grounded': result.get('is_grounded', True),
            'model': result.get('model', ''),
        })
    except Exception as e:
        return JsonResponse({
            'response': "Sorry, I couldn't process your question right now. Please try again.",
            'sources': [],
            'status': 'error',
            'is_grounded': True,
        })


# ─── 5. Visual Mind-Map Generator ─────────────────────────────────────────────

@student_required
def mindmap_hub(request):
    student = request.user
    materials = StudyMaterial.objects.filter(
        subject__semester=student.semester
    ).select_related('subject', 'uploaded_by')

    if student.department:
        materials = materials.filter(subject__semester__department=student.department)

    query = request.GET.get('q', '').strip()
    if query:
        materials = materials.filter(
            Q(title__icontains=query) | Q(subject__name__icontains=query) | Q(subject__subject_code__icontains=query)
        )

    sub_id = request.GET.get('sub', '').strip()
    if sub_id:
        materials = materials.filter(subject_id=sub_id)

    subjects = Subject.objects.filter(semester=student.semester)
    if student.department:
        subjects = subjects.filter(semester__department=student.department)

    return render(request, 'student_panel/mindmap_hub.html', {
        'materials': materials,
        'subjects': subjects,
        'selected_sub': sub_id,
        'query': query,
    })


@student_required
def mindmap_generator(request, pk):
    """
    Visual Mind-Map / Educational Infographic Engine:
    - Strictly bound to the specific uploaded study material.
    - Format-independent extraction (PDF, PPTX, DOCX, TXT).
    - Verifies student ownership/access security.
    - Caches the generated high-resolution PNG image on the material.
    - Supports on-demand 'Regenerate Mind Map'.
    """
    student = request.user
    material = get_object_or_404(
        StudyMaterial.objects.select_related('subject', 'subject__semester', 'subject__semester__department'),
        pk=pk
    )

    # 1. Security Check: Never trust material ID from browser
    if not check_student_material_access(student, material):
        raise PermissionDenied("Access Denied: You do not have permission to view this mind map.")

    is_regenerate = request.GET.get('regenerate') == '1' or (request.method == 'POST' and request.POST.get('action') == 'regenerate')
    mindmap_url = None
    error_message = None

    # 2. Existing Mind Map Check: if already generated and file exists on disk, reuse immediately!
    if not is_regenerate and material.mindmap_image and material.mindmap_image.name:
        try:
            if os.path.exists(material.mindmap_image.path):
                mindmap_url = material.mindmap_image.url
        except Exception:
            mindmap_url = None

    # 3. Generate Visual Infographic Image if needed
    if not mindmap_url:
        if not material.file:
            error_message = "No study material file is attached to generate a visual mind map."
        else:
            try:
                # Universal document extraction
                doc = extract_document(material.file, document_title=material.title)
                
                # Educational infographic synthesis
                api_key = get_gemini_api_key()
                summary_json = material.ai_summary_json
                if not summary_json or not summary_json.get('sections'):
                    try:
                        summarizer = AISummarizerService(api_key=api_key)
                        summary_text, summary_json, _ = summarizer.summarize_document(doc)
                        material.ai_summary = summary_text
                        material.ai_summary_json = summary_json
                        material.ai_summary_updated_at = timezone.now()
                        material.save(update_fields=['ai_summary', 'ai_summary_json', 'ai_summary_updated_at'])
                    except Exception:
                        summary_json = None

                infographic_data = extract_infographic_data(doc, api_key=api_key, summary_json=summary_json)
                
                # Render high-resolution PNG image file
                from django.conf import settings
                os.makedirs(os.path.join(settings.MEDIA_ROOT, 'mindmaps'), exist_ok=True)
                rel_filename = f"mindmaps/mindmap_{material.id}.png"
                abs_filepath = os.path.join(settings.MEDIA_ROOT, rel_filename)
                
                render_mindmap_image(infographic_data, abs_filepath)
                
                # Persist to StudyMaterial
                material.mindmap_image.name = rel_filename
                material.mindmap_updated_at = timezone.now()
                material.save(update_fields=['mindmap_image', 'mindmap_updated_at'])
                
                mindmap_url = material.mindmap_image.url
            except Exception as e:
                error_message = "Unable to generate the mind map. Please try again."

    return render(request, 'student_panel/mindmap_view.html', {
        'material': material,
        'mindmap_url': mindmap_url,
        'error_message': error_message,
        'is_cached': bool(not is_regenerate and material.mindmap_image),
        'updated_at': material.mindmap_updated_at,
    })

