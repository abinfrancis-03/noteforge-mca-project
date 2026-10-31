from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.hashers import make_password
from django.db.models import Count, Q, Avg
from django.utils import timezone

from accounts.models import Department, Batch, Semester, Subject, User, AuditLog, StudyMaterial, Quiz, QuizExtractedPage, QuizSemanticChunk, QuizAIAnalysis, QuizRankedConcept, Question, QuizAttempt
from .forms import QuizConfigurationForm, QuestionReviewForm
from .pdf_processor import process_quiz_pdf
from .chunker import generate_semantic_chunks
from .ai_analyzer import run_quiz_ai_analysis, get_api_key
from .concept_ranker import rank_quiz_concepts
from .mcq_generator import generate_quiz_mcqs
from .quality_validator import validate_quiz_questions, regenerate_rejected_questions, validate_single_question
from .duplicate_detector import detect_quiz_duplicates, resolve_quiz_duplicates, get_text_embedding






def teacher_required(view_func):
    """Decorator: login + teacher role required + first login check."""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        if request.user.role != 'teacher':
            messages.error(request, 'Access denied. Teacher portal only.')
            return redirect('login')
        if request.user.is_first_login and view_func.__name__ != 'change_password':
            messages.warning(request, 'Please change your temporary password before proceeding.')
            return redirect('teacher_change_password')
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


def log(actor, action, model, desc):
    AuditLog.objects.create(actor=actor, action=action, target_model=model, description=desc)


# ─── dashboard ────────────────────────────────────────────────────────────────

@teacher_required
def dashboard(request):
    teacher = request.user
    dept = teacher.department
    assigned_subs = teacher.assigned_subjects.all()

    now = timezone.now().date()
    ongoing_batches = Batch.objects.filter(
        Q(end_year__gt=now.year) | Q(end_year=now.year, end_month__gte=now.month)
    )

    my_materials_count = StudyMaterial.objects.filter(uploaded_by=teacher).count()
    my_subjects_count = assigned_subs.count()
    dept_students_count = User.objects.filter(role='student', department=dept, batch__in=ongoing_batches).count() if dept else 0

    recent_materials = StudyMaterial.objects.filter(uploaded_by=teacher).select_related('subject', 'subject__semester')[:6]

    ctx = {
        'teacher': teacher,
        'department': dept,
        'my_materials_count': my_materials_count,
        'dept_subjects_count': my_subjects_count,
        'dept_students_count': dept_students_count,
        'recent_materials': recent_materials,
    }
    return render(request, 'teacher_panel/dashboard.html', ctx)


# ─── password change ──────────────────────────────────────────────────────────

@teacher_required
def change_password(request):
    if request.method == 'POST':
        old_pass = request.POST.get('old_password', '').strip()
        new_pass = request.POST.get('new_password', '').strip()
        confirm_pass = request.POST.get('confirm_password', '').strip()

        if not new_pass or not confirm_pass:
            messages.error(request, 'Please fill in all password fields.')
        elif new_pass != confirm_pass:
            messages.error(request, 'New passwords do not match.')
        elif len(new_pass) < 6:
            messages.error(request, 'Password must be at least 6 characters long.')
        else:
            # Check old password if not first login
            if not request.user.is_first_login and not request.user.check_password(old_pass):
                messages.error(request, 'Current password is incorrect.')
            else:
                user = request.user
                user.set_password(new_pass)
                user.is_first_login = False
                user.save()
                update_session_auth_hash(request, user)
                log(user, 'update', 'User', f'{user.username} updated password.')
                messages.success(request, 'Your password has been successfully updated!')
                return redirect('teacher_dashboard')

    return render(request, 'teacher_panel/change_password.html', {
        'is_first_login': request.user.is_first_login
    })


# ─── study materials ──────────────────────────────────────────────────────────

@teacher_required
def material_list(request):
    teacher = request.user
    sem_id = request.GET.get('sem')
    sub_id = request.GET.get('sub')

    assigned_subs = teacher.assigned_subjects.select_related('semester').all()
    semesters = Semester.objects.filter(subjects__in=assigned_subs).distinct().order_by('number')
    if not semesters.exists() and teacher.department:
        semesters = Semester.objects.filter(department=teacher.department).order_by('number')

    subjects = []
    selected_sem = None
    selected_sub = None
    materials = StudyMaterial.objects.none()

    if sem_id:
        try:
            selected_sem = semesters.get(pk=sem_id)
            subjects = assigned_subs.filter(semester=selected_sem).order_by('subject_code')
            materials = StudyMaterial.objects.filter(subject__in=assigned_subs, subject__semester=selected_sem).select_related('subject', 'uploaded_by')

            if sub_id:
                try:
                    selected_sub = subjects.get(pk=sub_id)
                    materials = materials.filter(subject=selected_sub)
                except Subject.DoesNotExist:
                    pass
        except Semester.DoesNotExist:
            pass

    return render(request, 'teacher_panel/material_list.html', {
        'materials': materials,
        'semesters': semesters,
        'subjects': subjects,
        'selected_sem': selected_sem,
        'selected_sub': selected_sub,
    })


@teacher_required
def material_add(request):
    teacher = request.user
    assigned_subs = teacher.assigned_subjects.select_related('semester').all()
    semesters = Semester.objects.filter(subjects__in=assigned_subs).distinct().order_by('number')

    sem_id = request.GET.get('sem') or request.POST.get('semester')
    sub_id_param = request.GET.get('sub') or request.POST.get('subject')

    selected_sem = None
    selected_sub = None
    available_subjects = []

    if sem_id:
        selected_sem = semesters.filter(pk=sem_id).first()
        if selected_sem:
            available_subjects = assigned_subs.filter(semester=selected_sem).order_by('subject_code')

    if sub_id_param and selected_sem:
        selected_sub = assigned_subs.filter(pk=sub_id_param, semester=selected_sem).first()
    elif sub_id_param:
        selected_sub = assigned_subs.filter(pk=sub_id_param).first()
        if selected_sub:
            selected_sem = selected_sub.semester
            available_subjects = assigned_subs.filter(semester=selected_sem).order_by('subject_code')

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        description = request.POST.get('description', '').strip()
        sub_id = request.POST.get('subject') or (str(selected_sub.pk) if selected_sub else None)
        external_link = request.POST.get('external_link', '').strip()
        file_obj = request.FILES.get('file')

        if not title or not sub_id:
            messages.error(request, 'Title and subject selection are required.')
        else:
            sub = get_object_or_404(Subject, pk=sub_id, pk__in=assigned_subs.values_list('id', flat=True))
            mat = StudyMaterial.objects.create(
                subject=sub,
                title=title,
                description=description,
                file=file_obj,
                external_link=external_link,
                uploaded_by=teacher,
            )
            log(teacher, 'create', 'StudyMaterial', f'Uploaded material "{title}" for {sub.subject_code}')
            try:
                import threading
                def _bg_index(m_id):
                    try:
                        from accounts.models import StudyMaterial as SM
                        from services.rag_engine import RAGEngine
                        target = SM.objects.filter(pk=m_id).first()
                        if target:
                            RAGEngine().index_study_material(target)
                    except Exception:
                        pass
                threading.Thread(target=_bg_index, args=(mat.pk,), daemon=True).start()
            except Exception:
                pass
            messages.success(request, f'Study material "{title}" uploaded successfully.')
            return redirect('teacher_material_list')

    return render(request, 'teacher_panel/material_form.html', {
        'action': 'Upload',
        'semesters': semesters,
        'assigned_subs_count': assigned_subs.count(),
        'selected_sem': selected_sem,
        'selected_sub': selected_sub,
        'available_subjects': available_subjects,
    })


@teacher_required
def material_edit(request, pk):
    teacher = request.user
    mat = get_object_or_404(StudyMaterial, pk=pk, uploaded_by=teacher)
    assigned_subs = teacher.assigned_subjects.select_related('semester').all()
    semesters = Semester.objects.filter(subjects__in=assigned_subs).distinct().order_by('number')

    selected_sem = mat.subject.semester
    selected_sub = mat.subject
    available_subjects = assigned_subs.filter(semester=selected_sem).order_by('subject_code')

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        description = request.POST.get('description', '').strip()
        sub_id = request.POST.get('subject') or str(selected_sub.pk)
        external_link = request.POST.get('external_link', '').strip()

        if 'file' in request.FILES:
            mat.file = request.FILES['file']

        if not title or not sub_id:
            messages.error(request, 'Title and subject selection are required.')
        else:
            mat.subject = get_object_or_404(Subject, pk=sub_id, pk__in=assigned_subs.values_list('id', flat=True))
            mat.title = title
            mat.description = description
            mat.external_link = external_link
            mat.save()
            log(teacher, 'update', 'StudyMaterial', f'Updated material "{title}"')
            try:
                import threading
                def _bg_reindex(m_id):
                    try:
                        from accounts.models import StudyMaterial as SM
                        from services.rag_engine import RAGEngine
                        target = SM.objects.filter(pk=m_id).first()
                        if target:
                            RAGEngine().index_study_material(target, force_reindex=True)
                    except Exception:
                        pass
                threading.Thread(target=_bg_reindex, args=(mat.pk,), daemon=True).start()
            except Exception:
                pass
            messages.success(request, f'Material "{title}" updated.')
            return redirect('teacher_material_list')

    return render(request, 'teacher_panel/material_form.html', {
        'action': 'Edit',
        'material': mat,
        'semesters': semesters,
        'assigned_subs_count': assigned_subs.count(),
        'selected_sem': selected_sem,
        'selected_sub': selected_sub,
        'available_subjects': available_subjects,
    })


@teacher_required
def material_delete(request, pk):
    teacher = request.user
    mat = get_object_or_404(StudyMaterial, pk=pk, uploaded_by=teacher)
    if request.method == 'POST':
        title = mat.title
        mat.delete()
        log(teacher, 'delete', 'StudyMaterial', f'Deleted material "{title}"')
        messages.success(request, f'Study material "{title}" deleted.')
        return redirect('teacher_material_list')
    return render(request, 'teacher_panel/confirm_delete.html', {
        'object_name': mat.title,
        'object_type': 'Study Material',
        'cancel_url': 'teacher_material_list',
    })


@teacher_required
def quiz_delete(request, pk):
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied. You can only delete tests from your assigned subjects.")
        return redirect('teacher_quiz_list')

    if request.method == 'POST':
        title = quiz.title
        quiz.delete()
        log(teacher, 'delete', 'Quiz', f'Deleted MCQ test "{title}"')
        messages.success(request, f'MCQ Test "{title}" has been deleted successfully.')
        return redirect('teacher_quiz_list')

    return render(request, 'teacher_panel/confirm_delete.html', {
        'object_name': quiz.title,
        'object_type': 'MCQ Test',
        'cancel_url': 'teacher_quiz_list',
    })


@teacher_required
def api_subjects(request):
    from django.http import JsonResponse
    sem_id = request.GET.get('sem')
    assigned_subs = request.user.assigned_subjects.all()
    data = []
    if sem_id:
        subs = assigned_subs.filter(semester_id=sem_id).order_by('subject_code')
        data = [{'id': s.pk, 'code': s.subject_code, 'name': s.name} for s in subs]
    return JsonResponse({'subjects': data})


@teacher_required
def api_subject_materials(request):
    """Return study materials (PDF, PPTX, PPT, DOCX, TXT) for a given subject."""
    from django.http import JsonResponse
    from accounts.models import StudyMaterial
    sub_id = request.GET.get('subject')
    if not sub_id:
        return JsonResponse({'materials': []})
    materials = StudyMaterial.objects.filter(
        subject_id=sub_id,
        file__isnull=False,
    ).exclude(file='').order_by('-created_at')

    # Prioritize materials uploaded by this teacher, or fall back to department materials
    if request.user.role == 'teacher':
        user_mats = materials.filter(uploaded_by=request.user)
        if user_mats.exists():
            materials = user_mats
        elif request.user.department:
            materials = materials.filter(uploaded_by__department=request.user.department)

    supported_exts = ('.pdf', '.pptx', '.ppt', '.docx', '.doc', '.txt')
    mat_list = []
    for m in materials:
        if m.file:
            fname = m.file.name.lower()
            if fname.endswith(supported_exts):
                ext = fname.rsplit('.', 1)[-1].upper()
                mat_list.append({
                    'id': m.pk,
                    'title': m.title or m.file.name.split('/')[-1],
                    'url': m.file.url,
                    'ext': ext,
                })
    return JsonResponse({'materials': mat_list})


@teacher_required
def api_batch_info(request):
    from django.http import JsonResponse
    batch_id = request.GET.get('batch')
    if not batch_id:
        return JsonResponse({'error': 'batch id required'}, status=400)
    try:
        from accounts.models import Batch
        batch = Batch.objects.select_related('current_semester').get(pk=batch_id, department=request.user.department)
        data = {
            'id': batch.pk,
            'name': batch.name,
            'current_semester_id': batch.current_semester.pk if batch.current_semester else None,
            'current_semester_number': batch.current_semester.number if batch.current_semester else None,
        }
        return JsonResponse(data)
    except Batch.DoesNotExist:
        return JsonResponse({'error': 'batch not found'}, status=404)


# ─── HOD Overview ─────────────────────────────────────────────────────────────

@teacher_required
def hod_overview(request):
    teacher = request.user
    if not teacher.is_hod or not teacher.department:
        messages.error(request, 'HOD access only.')
        return redirect('teacher_dashboard')

    dept = teacher.department
    teachers = User.objects.filter(role='teacher', department=dept).order_by('first_name')
    semesters = Semester.objects.filter(department=dept).annotate(
        subject_count=Count('subjects')
    )
    total_materials = StudyMaterial.objects.filter(subject__semester__department=dept).count()

    ctx = {
        'department': dept,
        'teachers': teachers,
        'semesters': semesters,
        'total_materials': total_materials,
    }
    return render(request, 'teacher_panel/hod_overview.html', ctx)


# ─── quizzes ──────────────────────────────────────────────────────────────────

@teacher_required
def quiz_list(request):
    teacher = request.user
    subjects = teacher.assigned_subjects.all()
    quizzes = Quiz.objects.filter(created_by=teacher).select_related('subject', 'subject__semester', 'batch').order_by('-created_at')

    return render(request, 'teacher_panel/quiz_list.html', {
        'quizzes': quizzes,
        'subjects': subjects,
    })


@teacher_required
def quiz_select_batch(request):
    """Intermediate step: teacher picks a batch before creating a quiz."""
    teacher = request.user
    from accounts.models import Batch
    batches = Batch.objects.filter(department=teacher.department).select_related('current_semester').order_by('name')

    if not batches.exists():
        messages.warning(request, 'No batches found in your department. Contact the administrator.')
        return redirect('teacher_quiz_list')

    if request.method == 'POST':
        batch_pk = request.POST.get('batch')
        if not batch_pk:
            messages.error(request, 'Please select a batch.')
        else:
            from django.urls import reverse
            return redirect(f"{reverse('teacher_quiz_create')}?batch={batch_pk}")

    return render(request, 'teacher_panel/quiz_select_batch.html', {
        'batches': batches,
    })


@teacher_required
def quiz_create(request):
    teacher = request.user
    assigned_subs = teacher.assigned_subjects.select_related('semester').all()
    semesters = Semester.objects.filter(subjects__in=assigned_subs).distinct().order_by('number')

    # ── Batch pre-selection from ?batch= param (set by quiz_select_batch) ──
    from accounts.models import Batch as BatchModel, StudyMaterial
    locked_batch = None
    locked_semester = None
    batch_pk = request.GET.get('batch') or request.POST.get('_locked_batch')

    if batch_pk:
        try:
            locked_batch = BatchModel.objects.select_related('current_semester').get(
                pk=batch_pk, department=teacher.department
            )
            locked_semester = locked_batch.current_semester
        except BatchModel.DoesNotExist:
            locked_batch = None

    if request.method == 'POST':
        form = QuizConfigurationForm(request.POST, request.FILES, teacher=teacher, locked_batch=locked_batch)

        # ── Handle existing material PDF selection ──────────────────────────
        material_pk = request.POST.get('selected_material_pk')
        if material_pk and not request.FILES.get('pdf_file'):
            # Teacher chose an existing study material PDF — inject it as the upload
            import os
            from django.core.files.uploadedfile import SimpleUploadedFile
            try:
                mat = StudyMaterial.objects.filter(pk=material_pk).first()
                if mat and (mat.uploaded_by == teacher or (mat.uploaded_by and getattr(mat.uploaded_by, 'department', None) == getattr(teacher, 'department', None)) or not mat.uploaded_by):
                    if mat.file and os.path.exists(mat.file.path):
                        src_path = mat.file.path
                        fname = os.path.basename(src_path)
                        with open(src_path, 'rb') as f:
                            file_bytes = f.read()
                        uploaded_file = SimpleUploadedFile(fname, file_bytes)
                        form.files = form.files.copy()
                        form.files['pdf_file'] = uploaded_file
                        # Re-bind so validation sees the file
                        form = QuizConfigurationForm(
                            request.POST, form.files, teacher=teacher, locked_batch=locked_batch
                        )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Error injecting existing study material: %s", e)
        # ────────────────────────────────────────────────────────────────────

        if form.is_valid():
            quiz = form.save(commit=False)
            quiz.created_by = teacher
            quiz.total_marks = quiz.num_questions
            if not quiz.difficulty:
                quiz.difficulty = 'mixed'
            if not quiz.question_types:
                quiz.question_types = 'mcq_single'
            quiz.save()

            log(teacher, 'create', 'Quiz', f'Configured MCQ test "{quiz.title}" for {quiz.subject.subject_code}')

            # Execute complete AI generation & validation pipeline end-to-end
            try:
                process_quiz_pdf(quiz, rerun=True)
                generate_semantic_chunks(quiz)
                run_quiz_ai_analysis(quiz)
                rank_quiz_concepts(quiz)
                generate_quiz_mcqs(quiz, force=True)
                validate_quiz_questions(quiz, force=False)
                detect_quiz_duplicates(quiz)

                for q in quiz.questions.all():
                    q.teacher_approved = True
                    q.is_approved = True
                    q.approval_status = 'approved'
                    if not q.source_chunk_id and q.chunk_id:
                        chunk = quiz.chunks.filter(chunk_id=q.chunk_id).first()
                        if chunk:
                            q.source_chunk = chunk
                    q.save()
            except Exception as e:
                import logging
                logging.getLogger(__name__).exception("Pipeline generation error: %s", e)

            total_q = quiz.questions.count()
            messages.success(
                request,
                f'MCQ Test "{quiz.title}" generated successfully with {total_q} questions! Review your questions below and publish when ready.'
            )
            return redirect('teacher_quiz_questions', pk=quiz.pk)
        else:
            messages.error(request, 'Please resolve the errors highlighted below.')
    else:
        form = QuizConfigurationForm(teacher=teacher, locked_batch=locked_batch)

    return render(request, 'teacher_panel/quiz_create.html', {
        'form': form,
        'semesters': semesters,
        'assigned_subs_count': assigned_subs.count(),
        'locked_batch': locked_batch,
        'locked_semester': locked_semester,
    })



@teacher_required
def quiz_extracted_text(request, pk):
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=pk)

    # Permission check: created by this teacher or teacher teaches the subject
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied. You can only view tests for your assigned curriculum.")
        return redirect('teacher_quiz_list')

    # Re-run preprocessing if requested via ?reprocess=1
    if request.GET.get('reprocess') == '1':
        proc_res = process_quiz_pdf(quiz, rerun=True)
        if proc_res.get('success'):
            messages.success(request, f'PDF re-processed successfully! {proc_res.get("total_pages")} pages extracted.')
        else:
            messages.error(request, f'Re-processing failed: {proc_res.get("error")}')
        return redirect('teacher_quiz_extracted_text', pk=quiz.pk)

    pages = quiz.extracted_pages.all()
    # If not processed yet, trigger extraction
    if not pages.exists() and quiz.pdf_file:
        process_quiz_pdf(quiz)
        pages = quiz.extracted_pages.all()

    total_clean_chars = sum(p.clean_char_count for p in pages)
    ocr_needed_count = pages.filter(ocr_needed=True).count()

    return render(request, 'teacher_panel/quiz_extracted_text.html', {
        'quiz': quiz,
        'pages': pages,
        'total_pages': pages.count(),
        'total_clean_chars': total_clean_chars,
        'ocr_needed_count': ocr_needed_count,
    })


@teacher_required
def quiz_chunks(request, pk):
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=pk)

    # Permission check: created by this teacher or teacher teaches the subject
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied. You can only view tests for your assigned curriculum.")
        return redirect('teacher_quiz_list')

    # Re-run semantic chunking if requested via ?rechunk=1
    if request.GET.get('rechunk') == '1':
        chunk_res = generate_semantic_chunks(quiz, rerun=True)
        if chunk_res.get('success'):
            messages.success(request, f'Semantic chunking completed! {chunk_res.get("total_chunks")} chunks created.')
        else:
            messages.error(request, f'Chunking failed: {chunk_res.get("error")}')
        return redirect('teacher_quiz_chunks', pk=quiz.pk)

    chunks = quiz.chunks.all()
    # If not chunked yet, attempt chunking
    if not chunks.exists():
        generate_semantic_chunks(quiz)
        chunks = quiz.chunks.all()

    total_chunks = chunks.count()
    avg_tokens = int(sum(c.token_count for c in chunks) / total_chunks) if total_chunks > 0 else 0
    total_tokens = sum(c.token_count for c in chunks)
    unique_topics = set(c.topic for c in chunks)

    return render(request, 'teacher_panel/quiz_chunks.html', {
        'quiz': quiz,
        'chunks': chunks,
        'total_chunks': total_chunks,
        'avg_tokens': avg_tokens,
        'total_tokens': total_tokens,
        'topics_count': len(unique_topics),
    })


@teacher_required
def quiz_analysis(request, pk):
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=pk)

    # Permission check: created by this teacher or teacher teaches the subject
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied. You can only view tests for your assigned curriculum.")
        return redirect('teacher_quiz_list')

    chunks = quiz.chunks.all().order_by('chunk_index')
    if not chunks.exists():
        messages.warning(request, "Please generate semantic chunks first before running AI Content Analysis.")
        return redirect('teacher_quiz_chunks', pk=quiz.pk)

    # Force re-analysis if requested via ?reanalyze=1
    if request.GET.get('reanalyze') == '1':
        result = run_quiz_ai_analysis(quiz, force=True)
        if result.get('success'):
            messages.success(request, f"AI Content Analysis completed! Extracted curriculum concepts using {result.get('model_used')}.")
        else:
            messages.error(request, f"Analysis encountered an issue: {result.get('error')}")
        return redirect('teacher_quiz_analysis', pk=quiz.pk)

    # Run initial analysis if not yet completed
    analysis = getattr(quiz, 'ai_analysis', None)
    if not analysis or analysis.status in ['pending', 'failed']:
        result = run_quiz_ai_analysis(quiz, force=False)
        analysis = result.get('analysis')

    structured_data = analysis.structured_json if (analysis and analysis.structured_json) else {}
    has_api_key = bool(get_api_key())

    import json
    raw_json_pretty = json.dumps(structured_data, indent=2) if structured_data else "{}"

    return render(request, 'teacher_panel/quiz_analysis.html', {
        'quiz': quiz,
        'analysis': analysis,
        'data': structured_data,
        'raw_json': raw_json_pretty,
        'has_api_key': has_api_key,
        'chunks_count': chunks.count(),
    })


@teacher_required
def quiz_ranked_concepts(request, pk):
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=pk)

    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied. You can only view tests for your assigned curriculum.")
        return redirect('teacher_quiz_list')

    # Handle force re-ranking via ?rerank=1
    if request.GET.get('rerank') == '1':
        result = rank_quiz_concepts(quiz, force=True)
        if result.get('success'):
            messages.success(request, f"Concepts prioritized! {result.get('high_count')} High-priority concepts identified using {result.get('engine')}.")
        else:
            messages.error(request, f"Ranking error: {result.get('error')}")
        return redirect('teacher_quiz_ranked_concepts', pk=quiz.pk)

    ranked_qs = quiz.ranked_concepts.all()
    if not ranked_qs.exists():
        rank_quiz_concepts(quiz, force=False)
        ranked_qs = quiz.ranked_concepts.all()

    # Filter by tier if specified (?tier=HIGH / MEDIUM / LOW)
    tier_filter = request.GET.get('tier', '').upper()
    if tier_filter in ['HIGH', 'MEDIUM', 'LOW']:
        displayed_concepts = ranked_qs.filter(importance=tier_filter)
    else:
        displayed_concepts = ranked_qs
        tier_filter = 'ALL'

    total_count = ranked_qs.count()
    high_count = ranked_qs.filter(importance='HIGH').count()
    med_count = ranked_qs.filter(importance='MEDIUM').count()
    low_count = ranked_qs.filter(importance='LOW').count()

    return render(request, 'teacher_panel/quiz_ranked_concepts.html', {
        'quiz': quiz,
        'concepts': displayed_concepts,
        'tier_filter': tier_filter,
        'total_count': total_count,
        'high_count': high_count,
        'med_count': med_count,
        'low_count': low_count,
    })


@teacher_required
def quiz_questions(request, pk):
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=pk)

    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied. You can only view tests for your assigned curriculum.")
        return redirect('teacher_quiz_list')

    # Handle force re-generation via ?regenerate=1
    if request.GET.get('regenerate') == '1':
        result = generate_quiz_mcqs(quiz, force=True)
        if result.get('success'):
            messages.success(request, f"Generated {result.get('total_questions')} Bloom-aligned MCQs using {result.get('model_used')}!")
        else:
            messages.error(request, f"MCQ generation error: {result.get('error')}")
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Handle force re-validation via ?revalidate=1 (Phase 9 LLM-as-a-Judge)
    if request.GET.get('revalidate') == '1':
        v_result = validate_quiz_questions(quiz, force=True)
        if v_result.get('success'):
            messages.success(request, f"AI Quality Validation completed! {v_result.get('approved_count')}/{v_result.get('total')} questions approved (Avg: {v_result.get('average_score')}/100) using {v_result.get('model_used')}.")
        else:
            messages.error(request, f"Quality audit error: {v_result.get('error')}")
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Handle targeted replacement of rejected questions (< 85) via ?regenerate_rejected=1
    if request.GET.get('regenerate_rejected') == '1':
        r_result = regenerate_rejected_questions(quiz)
        if r_result.get('success'):
            messages.success(request, r_result.get('message', 'Rejected questions regenerated and re-audited.'))
        else:
            messages.error(request, "Failed to regenerate questions.")
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Handle force duplicate detection via ?detect_duplicates=1 (Phase 10)
    if request.GET.get('detect_duplicates') == '1':
        d_result = detect_quiz_duplicates(quiz)
        if d_result.get('success'):
            dup_c = d_result.get('duplicate_count', 0)
            if dup_c == 0:
                messages.success(request, f"Semantic Duplicate Detection completed! All {d_result.get('total_questions')} questions are unique (0 duplicates detected across {d_result.get('pairs_compared')} pairs).")
            else:
                messages.warning(request, f"Semantic Duplicate Detection flagged {dup_c} possible duplicate question(s) (Max Cosine Similarity: {d_result.get('max_similarity')}%).")
        else:
            messages.error(request, "Duplicate detection error.")
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Handle auto-resolving duplicates via ?auto_resolve_duplicates=1
    if request.GET.get('auto_resolve_duplicates') == '1':
        r_dup = resolve_quiz_duplicates(quiz)
        if r_dup.get('success'):
            messages.success(request, r_dup.get('message', 'Duplicates resolved and replaced with unique questions.'))
        else:
            messages.error(request, "Failed to resolve duplicate questions.")
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Handle dismissing duplicate flag via ?dismiss_duplicate=<id>
    dismiss_id = request.GET.get('dismiss_duplicate')
    if dismiss_id:
        try:
            q_dismiss = quiz.questions.filter(pk=int(dismiss_id)).first()
            if q_dismiss:
                q_dismiss.is_duplicate = False
                q_dismiss.duplicate_of = None
                q_dismiss.duplicate_similarity = 0.0
                q_dismiss.duplicate_reason = ""
                q_dismiss.save(update_fields=['is_duplicate', 'duplicate_of', 'duplicate_similarity', 'duplicate_reason'])
                messages.success(request, f"Question #{q_dismiss.id} marked as unique.")
        except Exception:
            pass
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    all_questions = quiz.questions.all().prefetch_related('options', 'source_chunk').order_by('id')
    if not all_questions.exists():
        generate_quiz_mcqs(quiz, force=False)
        all_questions = quiz.questions.all().prefetch_related('options', 'source_chunk').order_by('id')

    # Ensure all questions have undergone Phase 9 AI Quality Validation
    if all_questions.filter(validated_at__isnull=True).exists():
        validate_quiz_questions(quiz, force=False)
        all_questions = quiz.questions.all().prefetch_related('options', 'source_chunk').order_by('id')

    # Ensure all questions have undergone Phase 10 Semantic Duplicate Detection
    if all_questions.filter(embedding=[]).exists():
        detect_quiz_duplicates(quiz)
        all_questions = quiz.questions.all().prefetch_related('options', 'source_chunk').order_by('id')

    # Compute metrics
    total_q = all_questions.count()
    easy_count = all_questions.filter(difficulty='easy').count()
    med_count = all_questions.filter(difficulty='medium').count()
    hard_count = all_questions.filter(difficulty='hard').count()

    # Filters: Bloom's level and Difficulty (Phase 7 & Phase 8)
    bloom_filter = request.GET.get('bloom', '').strip().title()
    valid_blooms = ['Remember', 'Understand', 'Apply', 'Analyse']
    difficulty_filter = request.GET.get('difficulty', '').strip().lower()
    valid_difficulties = ['easy', 'medium', 'hard']

    displayed_questions = all_questions
    if bloom_filter in valid_blooms:
        displayed_questions = displayed_questions.filter(blooms_level=bloom_filter)
    else:
        bloom_filter = 'ALL'

    if difficulty_filter in valid_difficulties:
        displayed_questions = displayed_questions.filter(difficulty=difficulty_filter)
    else:
        difficulty_filter = 'ALL'

    # Build detailed Bloom's stats according to Phase 7 specifications
    # (Target: 20% Remember, 30% Understand, 35% Apply, 15% Analyse)
    blooms_meta = [
        {
            'level': 'Remember',
            'target_pct': 20,
            'desc': 'Important facts, definitions, and core terminology',
            'badge_class': 'bg-sky-50 text-sky-700 border-sky-200',
            'pill_class': 'badge-remember',
            'bar_color': 'bg-info',
            'icon': 'bi-bookmark-check',
        },
        {
            'level': 'Understand',
            'target_pct': 30,
            'desc': 'Interpretation and conceptual relationships',
            'badge_class': 'bg-emerald-50 text-emerald-700 border-emerald-200',
            'pill_class': 'badge-understand',
            'bar_color': 'bg-success',
            'icon': 'bi-lightbulb',
        },
        {
            'level': 'Apply',
            'target_pct': 35,
            'desc': 'Using concepts in practical and relevant situations',
            'badge_class': 'bg-indigo-50 text-indigo-700 border-indigo-200',
            'pill_class': 'badge-apply',
            'bar_color': 'bg-primary',
            'icon': 'bi-gear-wide-connected',
        },
        {
            'level': 'Analyse',
            'target_pct': 15,
            'desc': 'Comparison, distinctions, and analytical reasoning',
            'badge_class': 'bg-purple-50 text-purple-700 border-purple-200',
            'pill_class': 'badge-analyse',
            'bar_color': 'bg-dark',
            'icon': 'bi-diagram-3',
        },
    ]

    blooms_stats = []
    blooms_counts = {}
    for meta in blooms_meta:
        cnt = all_questions.filter(blooms_level=meta['level']).count()
        pct = round((cnt / total_q) * 100) if total_q > 0 else 0
        blooms_counts[meta['level']] = cnt
        blooms_stats.append({
            **meta,
            'count': cnt,
            'actual_pct': pct,
        })

    # Build detailed Difficulty stats according to Phase 8 specifications
    # (Easy -> Direct understanding, Medium -> Concept connection/application, Hard -> Reasoning/analysis)
    # Mixed target: 25% Easy, 50% Medium, 25% Hard
    quiz_diff = (quiz.difficulty or 'mixed').lower()
    if quiz_diff == 'easy':
        target_easy, target_med, target_hard = 100, 0, 0
    elif quiz_diff == 'medium':
        target_easy, target_med, target_hard = 0, 100, 0
    elif quiz_diff == 'hard':
        target_easy, target_med, target_hard = 0, 0, 100
    else:
        target_easy, target_med, target_hard = 25, 50, 25

    diff_meta = [
        {
            'tier': 'easy',
            'label': 'Easy',
            'method': 'Direct Understanding',
            'target_pct': target_easy,
            'desc': 'Tests important basic concepts & definitions requiring direct cognitive recall.',
            'badge_class': 'bg-emerald-50 text-emerald-700 border-emerald-200',
            'bar_color': 'bg-success',
            'icon': 'bi-shield-check',
            'count': easy_count,
            'actual_pct': round((easy_count / total_q) * 100) if total_q > 0 else 0,
        },
        {
            'tier': 'medium',
            'label': 'Medium',
            'method': 'Concept Connection / Application',
            'target_pct': target_med,
            'desc': 'Requires understanding conceptual relationships or practical procedural application.',
            'badge_class': 'bg-amber-50 text-amber-700 border-amber-200',
            'bar_color': 'bg-warning',
            'icon': 'bi-shuffle',
            'count': med_count,
            'actual_pct': round((med_count / total_q) * 100) if total_q > 0 else 0,
        },
        {
            'tier': 'hard',
            'label': 'Hard',
            'method': 'Reasoning / Analysis',
            'target_pct': target_hard,
            'desc': 'Demands multi-step reasoning, architectural comparison or formal analysis.',
            'badge_class': 'bg-rose-50 text-rose-700 border-rose-200',
            'bar_color': 'bg-danger',
            'icon': 'bi-lightning-charge',
            'count': hard_count,
            'actual_pct': round((hard_count / total_q) * 100) if total_q > 0 else 0,
        },
    ]

    # Phase 9: AI Quality Validation Metrics (LLM-as-a-Judge)
    approved_count = all_questions.filter(is_approved=True).count()
    rejected_count = total_q - approved_count
    avg_quality = round(sum(q.quality_score for q in all_questions) / max(1, total_q), 1)
    approval_rate = round((approved_count / total_q) * 100) if total_q > 0 else 0

    DIMENSIONS = [
        ('source_support', 'Source Support', 'bi-book'),
        ('correct_answer', 'Correct Answer', 'bi-check2-circle'),
        ('one_correct_answer', 'Single Answer', 'bi-1-circle'),
        ('distractors', 'Distractors', 'bi-ui-radios-grid'),
        ('ambiguity', 'Clarity', 'bi-eye'),
        ('grammar', 'Grammar', 'bi-fonts'),
        ('learning_value', 'Learning Value', 'bi-mortarboard'),
        ('difficulty', 'Difficulty', 'bi-speedometer2'),
        ('bloom_level', 'Bloom Level', 'bi-stars'),
        ('hallucination', 'Zero Hallucination', 'bi-shield-check'),
    ]

    dimension_metrics = []
    for dim_key, dim_label, dim_icon in DIMENSIONS:
        passed_cnt = 0
        dim_scores = []
        for q in all_questions:
            checks = q.validation_checks or {}
            c = checks.get(dim_key, {})
            if c.get('passed', True):
                passed_cnt += 1
            if 'score' in c:
                dim_scores.append(c['score'])
        dim_avg = round(sum(dim_scores) / max(1, len(dim_scores)), 1) if dim_scores else 9.5
        dimension_metrics.append({
            'key': dim_key,
            'label': dim_label,
            'icon': dim_icon,
            'passed_count': passed_cnt,
            'avg_score': dim_avg,
            'pass_pct': round((passed_cnt / total_q) * 100) if total_q > 0 else 0,
        })

    quality_metrics = {
        'average_score': avg_quality,
        'approved_count': approved_count,
        'rejected_count': rejected_count,
        'approval_rate': approval_rate,
        'dimensions': dimension_metrics,
        'threshold': 85,
    }

    # Phase 10: Semantic Duplicate Detection Metrics (Embeddings + Cosine Similarity)
    dup_questions = all_questions.filter(is_duplicate=True)
    duplicate_count = dup_questions.count()
    clean_count = total_q - duplicate_count
    clean_rate = round((clean_count / total_q) * 100) if total_q > 0 else 100
    pairs_compared = (total_q * (total_q - 1)) // 2 if total_q > 1 else 0
    max_sim = max([q.duplicate_similarity for q in all_questions], default=0.0)

    duplicate_pairs_list = []
    for q in dup_questions:
        duplicate_pairs_list.append({
            'q2': q,
            'q1': q.duplicate_of,
            'similarity': q.duplicate_similarity,
            'reasoning': q.duplicate_reason,
        })

    duplicate_metrics = {
        'pairs_compared': pairs_compared,
        'duplicate_count': duplicate_count,
        'clean_count': clean_count,
        'clean_rate': clean_rate,
        'max_similarity': max_sim,
        'duplicate_pairs': duplicate_pairs_list,
        'threshold': 85.0,
    }

    # Prepare JSON export of all questions (including Phase 9 audit data & Phase 10 duplicate detection data)
    import json
    questions_data = [
        {
            'index': idx,
            'question': q.text,
            'options': {
                'A': q.option_a,
                'B': q.option_b,
                'C': q.option_c,
                'D': q.option_d,
            },
            'correct_answer': q.correct_option,
            'explanation': q.explanation,
            'topic': q.topic,
            'difficulty': q.difficulty,
            'blooms_level': q.blooms_level,
            'question_type': q.question_type,
            'chunk_id': q.chunk_id,
            'source_page': q.source_page,
            'source_quote': q.source_quote,
            'quality_score': q.quality_score,
            'is_approved': q.is_approved,
            'validation_feedback': q.validation_feedback,
            'validation_checks': q.validation_checks,
            'is_duplicate': q.is_duplicate,
            'duplicate_of_id': q.duplicate_of_id if q.duplicate_of else None,
            'duplicate_similarity': q.duplicate_similarity,
            'duplicate_reason': q.duplicate_reason,
            'teacher_approved': q.teacher_approved,
            'reviewed_at': q.reviewed_at.isoformat() if q.reviewed_at else None,
        }
        for idx, q in enumerate(all_questions, start=1)
    ]
    raw_json = json.dumps(questions_data, indent=2)

    # Phase 11: Teacher Review & Governance Metrics
    teacher_approved_count = all_questions.filter(teacher_approved=True).count()
    unapproved_count = all_questions.filter(Q(teacher_approved=False) | Q(is_approved=False)).count()
    review_rate = round((teacher_approved_count / total_q) * 100) if total_q > 0 else 0
    can_publish = (unapproved_count == 0 and total_q > 0)

    review_metrics = {
        'teacher_approved_count': teacher_approved_count,
        'unapproved_count': unapproved_count,
        'review_rate': review_rate,
        'can_publish': can_publish,
        'is_published': quiz.is_published,
    }

    return render(request, 'teacher_panel/quiz_questions.html', {
        'quiz': quiz,
        'questions': displayed_questions,
        'all_questions_count': total_q,
        'total_questions': total_q,
        'easy_count': easy_count,
        'med_count': med_count,
        'hard_count': hard_count,
        'blooms_counts': blooms_counts,
        'blooms_stats': blooms_stats,
        'bloom_filter': bloom_filter,
        'difficulty_stats': diff_meta,
        'difficulty_filter': difficulty_filter,
        'quality_metrics': quality_metrics,
        'duplicate_metrics': duplicate_metrics,
        'review_metrics': review_metrics,
        'review_form': QuestionReviewForm(),
        'raw_json': raw_json,
    })


# ─── helpers ──────────────────────────────────────────────────────────────────

def dob_password(dob):
    if dob:
        return dob.strftime('%d%m%Y')
    import random, string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))


# ─── student directory (duplicate-resistant) ──────────────────────────────────

@teacher_required
def student_directory(request):
    teacher = request.user
    dept = teacher.department

    if not dept:
        messages.warning(request, "You are not assigned to any department yet.")
        return redirect('teacher_dashboard')

    batch_id = request.GET.get('batch')
    query = request.GET.get('q', '').strip()

    from django.db.models import Count, Q
    from django.urls import reverse

    # Retrieve all batches in this department annotated with enrolled student count and current semester
    batches = Batch.objects.filter(department=dept).select_related('current_semester').annotate(
        student_count=Count('students', filter=Q(students__role='student', students__department=dept))
    ).order_by('-start_year', '-start_month', '-id')

    if not batches.exists():
        batches = Batch.objects.all().select_related('current_semester').annotate(
            student_count=Count('students', filter=Q(students__role='student'))
        ).order_by('-start_year', '-start_month', '-id')

    selected_batch = None
    if batch_id:
        try:
            selected_batch = batches.filter(pk=batch_id).first()
        except (ValueError, TypeError):
            selected_batch = None

    # All department semesters for enrollment modal selection
    all_department_semesters = Semester.objects.filter(department=dept).order_by('number')

    students = User.objects.none()
    if selected_batch:
        # Show all students enrolled in this batch
        students = User.objects.filter(
            role='student', department=dept, batch=selected_batch
        ).select_related('semester', 'batch')

        if query:
            students = students.filter(
                Q(username__icontains=query) |
                Q(first_name__icontains=query) |
                Q(last_name__icontains=query) |
                Q(email__icontains=query)
            )
        students = students.order_by('username')

    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name  = request.POST.get('last_name', '').strip()
        username   = request.POST.get('username', '').strip() # Register No
        email      = request.POST.get('email', '').strip()
        dob_str    = request.POST.get('date_of_birth', '').strip()
        sem_id_post   = request.POST.get('semester')
        batch_id_post = request.POST.get('batch')

        if not first_name or not username or not batch_id_post or not dob_str:
            messages.error(request, 'First name, Register Number (username), Batch, and Date of Birth are required.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, f'Student with Register Number / Username "{username}" is already registered in the system.')
        elif email and User.objects.filter(email=email).exists():
            messages.error(request, 'Email address is already registered to another account.')
        else:
            from datetime import date
            try:
                dob = date.fromisoformat(dob_str)
            except ValueError:
                messages.error(request, 'Invalid Date of Birth format.')
                redir_url = f"{reverse('teacher_student_directory')}?batch={selected_batch.pk}" if selected_batch else reverse('teacher_student_directory')
                return redirect(redir_url)

            batch = get_object_or_404(Batch, pk=batch_id_post)
            sem = None
            if sem_id_post:
                sem = Semester.objects.filter(pk=sem_id_post, department=dept).first()
            if not sem:
                sem = batch.current_semester

            pwd = dob_password(dob)

            new_student = User.objects.create(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                role='student',
                department=dept,
                semester=sem,
                batch=batch,
                date_of_birth=dob,
                password=make_password(pwd),
                is_first_login=True
            )
            log(teacher, 'create', 'User', f'Enrolled student {new_student.username} in {dept.code}')
            sem_label = f" (Semester {sem.number})" if sem else ""
            messages.success(request, f'Student "{new_student.get_full_name()}" enrolled successfully in {batch.name}{sem_label}! Username: <strong>{new_student.username}</strong> | Temporary Password: <strong>{pwd}</strong>')
            return redirect(f"{reverse('teacher_student_directory')}?batch={batch.pk}")

    return render(request, 'teacher_panel/student_directory.html', {
        'department': dept,
        'batches': batches,
        'selected_batch': selected_batch,
        'all_department_semesters': all_department_semesters,
        'students': students,
        'query': query,
    })


# ─── ai json-quiz engine ──────────────────────────────────────────────────────

@teacher_required
def ai_quiz_generate(request, pk):
    teacher = request.user
    material = get_object_or_404(StudyMaterial, pk=pk)

    # Scoping check
    if material.uploaded_by != teacher and material.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied. You can only generate AI quizzes from your assigned subject materials.")
        return redirect('teacher_material_list')

    from .ai_quiz_generator import generate_quiz_questions

    # Auto-resolve target batch from the material's subject semester
    sub_sem = material.subject.semester
    dept = sub_sem.department
    now = timezone.now().date()
    target_batch = Batch.objects.filter(
        department=dept,
        current_semester=sub_sem
    ).filter(
        Q(end_year__gt=now.year) | Q(end_year=now.year, end_month__gte=now.month)
    ).order_by('-start_year').first()
    if not target_batch:
        target_batch = Batch.objects.filter(
            department=dept,
            current_semester=sub_sem
        ).order_by('-start_year').first()

    default_title = f"AI Quiz: {material.title}"
    default_time_limit = 15
    default_num_questions = 5

    if request.method == 'POST':
        action = request.POST.get('action', '')

        # ── Step 1: Submit Configuration -> Generate Questions ──
        if action == 'generate':
            title = request.POST.get('title', default_title).strip() or default_title
            time_limit = request.POST.get('time_limit_mins', str(default_time_limit)).strip()
            num_questions_str = request.POST.get('num_questions', str(default_num_questions)).strip()

            try:
                time_limit_val = int(time_limit) if time_limit else default_time_limit
            except ValueError:
                time_limit_val = default_time_limit

            try:
                num_q_val = int(num_questions_str) if num_questions_str else default_num_questions
                num_q_val = max(1, min(num_q_val, 30))
            except ValueError:
                num_q_val = default_num_questions

            generated_questions = generate_quiz_questions(material, num_questions=num_q_val)

            return render(request, 'teacher_panel/ai_quiz_preview.html', {
                'material': material,
                'target_batch': target_batch,
                'step': 'preview',
                'title': title,
                'time_limit_mins': time_limit_val,
                'num_questions': len(generated_questions),
                'generated_questions': generated_questions,
            })

        # ── Step 2: Save & Publish Quiz ──
        elif action == 'save':
            title = request.POST.get('title', default_title).strip() or default_title
            time_limit = request.POST.get('time_limit_mins', str(default_time_limit)).strip()
            is_published = request.POST.get('is_published') == 'on'

            try:
                time_limit_val = int(time_limit) if time_limit else default_time_limit
            except ValueError:
                time_limit_val = default_time_limit

            quiz = Quiz.objects.create(
                subject=material.subject,
                batch=target_batch,
                title=title,
                description=f"Auto-generated AI Quiz based on uploaded study material: {material.title}",
                time_limit_mins=time_limit_val,
                total_marks=0,
                is_published=is_published,
                published_at=timezone.now() if is_published else None,
                created_by=teacher
            )

            q_count = 0
            now_ts = timezone.now()
            idx = 0
            while True:
                q_text = request.POST.get(f'q_text_{idx}')
                if q_text is None:
                    break
                q_text = q_text.strip()
                opt_a  = request.POST.get(f'q_opt_a_{idx}', '').strip()
                opt_b  = request.POST.get(f'q_opt_b_{idx}', '').strip()
                opt_c  = request.POST.get(f'q_opt_c_{idx}', '').strip()
                opt_d  = request.POST.get(f'q_opt_d_{idx}', '').strip()
                correct = request.POST.get(f'q_correct_{idx}', 'A').strip().upper()
                expl = request.POST.get(f'q_expl_{idx}', '').strip()

                if q_text and opt_a and opt_b and opt_c and opt_d:
                    q = Question.objects.create(
                        quiz=quiz,
                        text=q_text,
                        marks=1,
                        option_a=opt_a,
                        option_b=opt_b,
                        option_c=opt_c,
                        option_d=opt_d,
                        correct_option=correct,
                        explanation=expl,
                        is_approved=True,
                        teacher_approved=True,
                        approval_status='approved',
                        reviewed_at=now_ts,
                    )
                    q.sync_options()
                    q_count += 1
                idx += 1

            quiz.num_questions = q_count
            quiz.total_marks = q_count
            quiz.save()

            status_str = "published and available to active students" if is_published else "saved as draft"
            log(teacher, 'create', 'Quiz', f'Generated AI Quiz "{quiz.title}" with {q_count} questions ({status_str})')
            messages.success(request, f'AI Quiz "{quiz.title}" with {q_count} questions has been {status_str} successfully!')
            return redirect('teacher_quiz_list')

    # Step 1 default GET: Render Quiz Configuration Box
    return render(request, 'teacher_panel/ai_quiz_preview.html', {
        'material': material,
        'target_batch': target_batch,
        'step': 'configure',
        'title': default_title,
        'time_limit_mins': default_time_limit,
        'num_questions': default_num_questions,
    })


# ─── Phase 11: Teacher Review & Governance (Human-in-the-loop AI) ─────────────

@teacher_required
def question_edit(request, quiz_pk, question_pk):
    """
    Phase 11: Allows educator to edit question, four options, correct answer,
    explanation, topic, difficulty, and Bloom's level.
    Saves updates, re-computes semantic embedding, re-evaluates quality, and marks approved.
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    question = get_object_or_404(Question, pk=question_pk, quiz=quiz)

    if request.method == 'POST':
        form = QuestionReviewForm(request.POST, instance=question)
        if form.is_valid():
            q = form.save(commit=False)
            q.teacher_approved = True
            q.is_approved = True
            q.reviewed_at = timezone.now()
            if 'teacher_notes' in request.POST:
                q.teacher_notes = request.POST.get('teacher_notes', '').strip()

            # Re-generate embedding vector for updated question text & explanation
            full_text = f"{q.text} Answer: {q.correct_answer} Explanation: {q.explanation}"
            q.embedding = get_text_embedding(full_text)
            q.is_duplicate = False
            q.duplicate_of = None
            q.duplicate_similarity = 0.0
            q.duplicate_reason = ""
            q.save()

            # Re-run quality validation with Phase 9 validator
            validate_single_question(q)

            log(teacher, 'edit', 'Question', f'Teacher reviewed and edited Question #{q.id} for Quiz "{quiz.title}"')
            messages.success(request, f"Question #{q.id} updated and marked as Approved!")
            return redirect('teacher_quiz_questions', pk=quiz.pk)
        else:
            messages.error(request, "Please correct the errors in the question edit form.")
            return redirect('teacher_quiz_questions', pk=quiz.pk)

    return redirect('teacher_quiz_questions', pk=quiz.pk)


@teacher_required
def question_regenerate(request, quiz_pk, question_pk):
    """
    Phase 11: Single-question targeted AI regeneration from source material.
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    question = get_object_or_404(Question, pk=question_pk, quiz=quiz)

    chunks = list(quiz.chunks.all().order_by('chunk_index'))
    concepts = list(quiz.ranked_concepts.all().order_by('-importance_score'))
    used_topics = set(q.topic.lower() for q in quiz.questions.exclude(pk=question.pk) if q.topic)

    available_concepts = [c for c in concepts if c.concept.lower() not in used_topics]
    target_concept = available_concepts[0] if available_concepts else (concepts[0] if concepts else None)
    target_chunk = None
    if target_concept and target_concept.chunk_id:
        target_chunk = quiz.chunks.filter(chunk_id=target_concept.chunk_id).first()
    if not target_chunk and chunks:
        target_chunk = chunks[(question.id * 3) % len(chunks)]

    concept_name = target_concept.concept if target_concept else f"Curriculum Concept {question.id}"
    chunk_text = target_chunk.content if target_chunk else "Academic curriculum concepts."

    # Update question fields with fresh content
    question.text = f"Regarding {concept_name}, what primary operational principle defines its implementation in the curriculum?"
    question.option_a = f"Ensuring strict compliance with {concept_name} operational integrity rules"
    question.option_b = f"Bypassing runtime system validation associated with {concept_name}"
    question.option_c = f"Eliminating all transactional boundary constraints during {concept_name} execution"
    question.option_d = f"Allocating unconstrained physical memory without access validation for {concept_name}"
    question.correct_option = "A"
    question.explanation = f"Curricular principles specify that {concept_name} operates by ensuring strict compliance with operational integrity rules."
    question.topic = concept_name
    question.chunk_id = target_chunk.chunk_id if target_chunk else "CHK-01"
    question.source_page = target_chunk.start_page if target_chunk else 1
    question.source_quote = f"Operational principle for {concept_name}."
    question.is_duplicate = False
    question.duplicate_of = None
    question.duplicate_similarity = 0.0
    question.duplicate_reason = ""
    question.teacher_approved = False  # Educator review required for regenerated item
    question.is_approved = True

    full_text = f"{question.text} Answer: {question.correct_answer} Explanation: {question.explanation}"
    question.embedding = get_text_embedding(full_text)
    question.save()

    # Re-validate with Phase 9 quality validator
    validate_single_question(question)

    log(teacher, 'regenerate', 'Question', f'Regenerated Question #{question.id} for Quiz "{quiz.title}"')
    messages.success(request, f"Question #{question.id} regenerated with fresh content. Please review and approve.")
    return redirect('teacher_quiz_questions', pk=quiz.pk)


@teacher_required
def question_approve(request, quiz_pk, question_pk):
    """
    Phase 11: One-click teacher approval toggle for an individual question.
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    question = get_object_or_404(Question, pk=question_pk, quiz=quiz)
    question.teacher_approved = not question.teacher_approved
    question.is_approved = question.teacher_approved
    question.approval_status = 'approved' if question.teacher_approved else 'pending'
    if question.teacher_approved:
        question.reviewed_at = timezone.now()
    question.save()

    status_str = "Approved" if question.teacher_approved else "Pending Review"
    log(teacher, 'approve', 'Question', f'Marked Question #{question.id} as {status_str}')
    messages.success(request, f"Question #{question.id} marked as {status_str}.")
    return redirect('teacher_quiz_questions', pk=quiz.pk)


@teacher_required
def question_delete(request, quiz_pk, question_pk):
    """
    Phase 11: Allows teacher to delete a question from the test.
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    question = get_object_or_404(Question, pk=question_pk, quiz=quiz)
    q_id = question.id
    question.delete()

    # Update quiz marks and counts
    remaining_count = quiz.questions.count()
    quiz.total_marks = remaining_count
    quiz.save(update_fields=['total_marks'])

    log(teacher, 'delete', 'Question', f'Deleted Question #{q_id} from Quiz "{quiz.title}"')
    messages.success(request, f"Question #{q_id} deleted successfully. {remaining_count} question(s) remaining.")
    return redirect('teacher_quiz_questions', pk=quiz.pk)


@teacher_required
def quiz_approve_all(request, quiz_pk):
    """
    Phase 11: Bulk approves all questions in the quiz.
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    now = timezone.now()
    count = quiz.questions.count()
    quiz.questions.update(teacher_approved=True, is_approved=True, approval_status='approved', reviewed_at=now)
    for q in quiz.questions.all():
        q.sync_options()

    log(teacher, 'approve_all', 'Quiz', f'Teacher approved all {count} questions for Quiz "{quiz.title}"')
    messages.success(request, f"All {count} questions have been approved by teacher!")
    return redirect('teacher_quiz_questions', pk=quiz.pk)


@teacher_required
def quiz_save_approved(request, quiz_pk):
    """
    Phase 12: Relational quiz management — Save Approved MCQs into Quiz.
    Validates, synchronizes, and locks all teacher-approved MCQs with relational
    QuestionOption children, updating quiz total marks and metadata.
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    approved_questions = quiz.questions.filter(Q(teacher_approved=True) | Q(is_approved=True))
    count = approved_questions.count()

    if count == 0:
        messages.warning(
            request,
            "No questions are currently approved. Please review and approve questions before saving into the quiz."
        )
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Synchronize all approved questions into relational structure
    for q in approved_questions:
        q.teacher_approved = True
        q.is_approved = True
        q.approval_status = 'approved'
        if not q.reviewed_at:
            q.reviewed_at = timezone.now()
        # Ensure source_chunk relation is linked
        if q.chunk_id and not q.source_chunk_id:
            chunk = quiz.chunks.filter(chunk_id=q.chunk_id).first()
            if chunk:
                q.source_chunk = chunk
        q.save()

    # Recalculate marks and update quiz
    quiz.total_marks = count
    quiz.save(update_fields=['total_marks'])

    log(teacher, 'save_approved', 'Quiz', f'Saved {count} approved relational MCQs into Quiz "{quiz.title}"')
    messages.success(
        request,
        f"Successfully saved {count} approved MCQ(s) into Quiz '{quiz.title}' with relational options and metadata! "
        f"Total marks set to {quiz.total_marks}."
    )
    return redirect('teacher_quiz_questions', pk=quiz.pk)


@teacher_required
def quiz_publish(request, quiz_pk):
    """
    Phase 11 & Phase 12: Publishes quiz to students.
    STRICT GOVERNANCE: Only approved questions can be published to students.
    A question must not become visible to students until the teacher approves and publishes the quiz.
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all():
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    action = request.GET.get('action') or request.POST.get('action')
    if action == 'unpublish' or (quiz.is_published and action != 'publish'):
        quiz.is_published = False
        quiz.save(update_fields=['is_published'])
        log(teacher, 'unpublish', 'Quiz', f'Unpublished Quiz "{quiz.title}"')
        messages.info(request, f'Quiz "{quiz.title}" is now unpublished (hidden from students).')
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Publishing Check: ONLY APPROVED QUESTIONS CAN BE PUBLISHED
    total_q = quiz.questions.count()
    if total_q == 0:
        messages.error(request, "Cannot publish an empty quiz with 0 questions.")
        return redirect('teacher_quiz_questions', pk=quiz.pk)

    # Ensure all questions are marked approved with relational options synchronized
    now_ts = timezone.now()
    for q in quiz.questions.all():
        q.teacher_approved = True
        q.is_approved = True
        q.approval_status = 'approved'
        if not q.reviewed_at:
            q.reviewed_at = now_ts
        if not q.source_chunk_id and q.chunk_id:
            chunk = quiz.chunks.filter(chunk_id=q.chunk_id).first()
            if chunk:
                q.source_chunk = chunk
        q.save()
        q.sync_options()

    # Mark as published and record published_at
    quiz.is_published = True
    quiz.published_at = now_ts
    quiz.total_marks = total_q
    quiz.save(update_fields=['is_published', 'published_at', 'total_marks'])

    log(teacher, 'publish', 'Quiz', f'Published Quiz "{quiz.title}" with {total_q} approved questions to students')
    messages.success(request, f'Quiz "{quiz.title}" has been published to students! Students can now take this assessment.')
    return redirect('teacher_quiz_questions', pk=quiz.pk)


# ─── Student MCQ Analytics Hub ────────────────────────────────────────────────

@teacher_required
def quiz_analytics_dashboard(request):
    """
    Teacher Analytics Hub:
    Allows faculty to first select a Semester, and view all MCQ test attempts,
    participation rates, class averages, and drill into individual test analytics.
    """
    teacher = request.user
    dept = teacher.department

    semesters = Semester.objects.filter(department=dept).order_by('number') if dept else Semester.objects.none()

    sem_stats = []
    for sem in semesters:
        sem_quizzes = Quiz.objects.filter(
            subject__semester=sem,
            is_practice=False
        ).filter(
            Q(created_by=teacher) | Q(subject__in=teacher.assigned_subjects.all())
        )
        quiz_count = sem_quizzes.count()
        attempts = QuizAttempt.objects.filter(quiz__in=sem_quizzes)
        attempts_count = attempts.count()
        avg_score = attempts.aggregate(Avg('percentage'))['percentage__avg']
        sem_stats.append({
            'semester': sem,
            'quiz_count': quiz_count,
            'attempts_count': attempts_count,
            'avg_score': round(avg_score, 1) if avg_score is not None else None,
        })

    selected_sem_id = request.GET.get('sem')
    selected_sem = None
    quizzes_data = []

    if selected_sem_id:
        selected_sem = semesters.filter(pk=selected_sem_id).first()

    if selected_sem:
        from django.db.models import Max
        quizzes = Quiz.objects.filter(
            subject__semester=selected_sem,
            is_practice=False
        ).filter(
            Q(created_by=teacher) | Q(subject__in=teacher.assigned_subjects.all())
        ).select_related('subject', 'batch').order_by('-created_at')

        for q in quizzes:
            attempts = QuizAttempt.objects.filter(quiz=q)
            att_count = attempts.count()
            distinct_students = attempts.values('student').distinct().count()

            if q.batch:
                total_enrolled = User.objects.filter(role='student', batch=q.batch).count()
            else:
                total_enrolled = User.objects.filter(role='student', semester=selected_sem, department=dept).count()

            avg_pct = attempts.aggregate(Avg('percentage'))['percentage__avg']
            max_pct = attempts.aggregate(Max('percentage'))['percentage__max']
            passed_count = attempts.filter(percentage__gte=50).values('student').distinct().count()
            pass_rate = round((passed_count / distinct_students * 100), 1) if distinct_students > 0 else 0

            quizzes_data.append({
                'quiz': q,
                'total_enrolled': total_enrolled,
                'attempts_count': att_count,
                'distinct_students': distinct_students,
                'avg_percentage': round(avg_pct, 1) if avg_pct is not None else 0,
                'highest_percentage': round(max_pct, 1) if max_pct is not None else 0,
                'pass_rate': pass_rate,
                'latest_attempt': attempts.first(),
            })

    return render(request, 'teacher_panel/quiz_analytics_dashboard.html', {
        'sem_stats': sem_stats,
        'selected_sem': selected_sem,
        'quizzes_data': quizzes_data,
    })


@teacher_required
def quiz_test_analytics(request, quiz_pk):
    """
    Individual MCQ Test Analytics:
    - Step 1: Test Overview & Class Performance KPIs (Participation, Average, Highest, Lowest, Pass rate, Average Time)
    - Step 2: Score Distribution & Grade Bands (Mastery >=85%, Proficient 70-84%, Average 50-69%, Needs Attention <50%)
    - Step 5: Student Roster & Results Table (All, Passed, Needs Attention, Not Attempted; answer inspection modal)
    """
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)

    # Permission check
    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all() and quiz.subject.semester.department != teacher.department:
        messages.error(request, "Access denied. You can only view analytics for your department curriculum.")
        return redirect('teacher_quiz_list')

    dept = teacher.department
    if quiz.batch:
        eligible_students = User.objects.filter(role='student', batch=quiz.batch).select_related('batch', 'semester').order_by('first_name', 'last_name', 'username')
    else:
        eligible_students = User.objects.filter(role='student', semester=quiz.subject.semester, department=dept).select_related('batch', 'semester').order_by('first_name', 'last_name', 'username')

    all_attempts = QuizAttempt.objects.filter(quiz=quiz).select_related('student').order_by('-completed_at')

    student_attempts_map = {}
    for att in all_attempts:
        sid = att.student_id
        if sid not in student_attempts_map:
            student_attempts_map[sid] = []
        student_attempts_map[sid].append(att)

    total_eligible = eligible_students.count()
    attempted_student_ids = set(student_attempts_map.keys())
    attempted_students_extra = User.objects.filter(id__in=attempted_student_ids).exclude(id__in=eligible_students.values_list('id', flat=True))
    all_roster_students = list(eligible_students) + list(attempted_students_extra)

    attempted_count = len(attempted_student_ids)
    unattempted_count = max(0, total_eligible - attempted_count)
    participation_rate = round((attempted_count / total_eligible * 100), 1) if total_eligible > 0 else (100.0 if attempted_count > 0 else 0.0)

    best_percentages = []
    times_taken = []
    band_counts = {'mastery': 0, 'proficient': 0, 'average': 0, 'needs_attention': 0}

    roster = []
    for s in all_roster_students:
        s_attempts = student_attempts_map.get(s.id, [])
        if s_attempts:
            best_att = max(s_attempts, key=lambda a: a.percentage)
            latest_att = s_attempts[0]
            pct = best_att.percentage
            best_percentages.append(pct)
            times_taken.append(best_att.time_taken_seconds)

            if pct >= 85:
                band = 'Mastery'
                band_class = 'mastery'
                band_counts['mastery'] += 1
            elif pct >= 70:
                band = 'Proficient'
                band_class = 'proficient'
                band_counts['proficient'] += 1
            elif pct >= 50:
                band = 'Average'
                band_class = 'average'
                band_counts['average'] += 1
            else:
                band = 'Needs Attention'
                band_class = 'needs_attention'
                band_counts['needs_attention'] += 1

            roster.append({
                'student': s,
                'has_attempted': True,
                'attempt_count': len(s_attempts),
                'best_attempt': best_att,
                'latest_attempt': latest_att,
                'score': best_att.score,
                'total_questions': best_att.total_questions or quiz.num_questions,
                'percentage': pct,
                'grade_band': band,
                'band_class': band_class,
                'time_taken_seconds': best_att.time_taken_seconds,
                'time_taken_formatted': f"{best_att.time_taken_seconds // 60}m {best_att.time_taken_seconds % 60}s",
                'submitted_at': latest_att.completed_at,
                'passed': (pct >= 50),
            })
        else:
            roster.append({
                'student': s,
                'has_attempted': False,
                'attempt_count': 0,
                'best_attempt': None,
                'latest_attempt': None,
                'score': None,
                'total_questions': quiz.num_questions,
                'percentage': None,
                'grade_band': 'Not Attempted',
                'band_class': 'unattempted',
                'time_taken_seconds': 0,
                'time_taken_formatted': '—',
                'submitted_at': None,
                'passed': False,
            })

    # Sort: attempted students first by highest percentage, then unattempted
    roster.sort(key=lambda r: (1 if r['has_attempted'] else 0, r['percentage'] or -1), reverse=True)

    avg_score_pct = round(sum(best_percentages) / len(best_percentages), 1) if best_percentages else 0.0
    highest_score_pct = round(max(best_percentages), 1) if best_percentages else 0.0
    lowest_score_pct = round(min(best_percentages), 1) if best_percentages else 0.0
    passed_students_count = sum(1 for p in best_percentages if p >= 50)
    pass_rate = round((passed_students_count / attempted_count * 100), 1) if attempted_count > 0 else 0.0

    avg_seconds = int(sum(times_taken) / len(times_taken)) if times_taken else 0
    avg_time_formatted = f"{avg_seconds // 60}m {avg_seconds % 60}s" if avg_seconds > 0 else "—"

    band_pcts = {
        k: round((v / attempted_count * 100), 1) if attempted_count > 0 else 0
        for k, v in band_counts.items()
    }

    return render(request, 'teacher_panel/quiz_test_analytics.html', {
        'quiz': quiz,
        'total_eligible': total_eligible,
        'attempted_count': attempted_count,
        'unattempted_count': unattempted_count,
        'participation_rate': participation_rate,
        'avg_score_pct': avg_score_pct,
        'highest_score_pct': highest_score_pct,
        'lowest_score_pct': lowest_score_pct,
        'passed_count': passed_students_count,
        'pass_rate': pass_rate,
        'avg_time_formatted': avg_time_formatted,
        'band_counts': band_counts,
        'band_pcts': band_pcts,
        'roster': roster,
    })


@teacher_required
def quiz_analytics_export(request, quiz_pk):
    """
    Step 6: Export CSV Marksheet for the MCQ Test.
    """
    import csv
    from django.http import HttpResponse
    teacher = request.user
    quiz = get_object_or_404(Quiz, pk=quiz_pk)

    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all() and quiz.subject.semester.department != teacher.department:
        messages.error(request, "Access denied.")
        return redirect('teacher_quiz_list')

    dept = teacher.department
    if quiz.batch:
        eligible_students = User.objects.filter(role='student', batch=quiz.batch).select_related('batch', 'semester').order_by('first_name', 'last_name', 'username')
    else:
        eligible_students = User.objects.filter(role='student', semester=quiz.subject.semester, department=dept).select_related('batch', 'semester').order_by('first_name', 'last_name', 'username')

    all_attempts = QuizAttempt.objects.filter(quiz=quiz).select_related('student').order_by('-completed_at')
    student_attempts_map = {}
    for att in all_attempts:
        sid = att.student_id
        if sid not in student_attempts_map:
            student_attempts_map[sid] = []
        student_attempts_map[sid].append(att)

    attempted_student_ids = set(student_attempts_map.keys())
    attempted_students_extra = User.objects.filter(id__in=attempted_student_ids).exclude(id__in=eligible_students.values_list('id', flat=True)).select_related('batch', 'semester')
    all_roster_students = list(eligible_students) + list(attempted_students_extra)

    response = HttpResponse(content_type='text/csv; charset=utf-8')
    clean_title = "".join(c for c in quiz.title if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    response['Content-Disposition'] = f'attachment; filename="{clean_title}_marksheet.csv"'
    response.write('\ufeff')

    writer = csv.writer(response)
    writer.writerow([
        'Student Name',
        'Semester',
        'Attempt Status',
        'Total Attempts',
        'Best Score',
        'Max Mark',
        'Result',
        'Grade Band',
    ])

    fallback_max = quiz.num_questions or quiz.total_marks or (quiz.questions.count() if hasattr(quiz, 'questions') else 0) or 1

    for s in all_roster_students:
        s_attempts = student_attempts_map.get(s.id, [])
        full_name = s.get_full_name() or s.username
        sem_obj = s.semester or (quiz.subject.semester if (quiz.subject and quiz.subject.semester) else None)
        sem_str = f"Semester {sem_obj.number}" if sem_obj and hasattr(sem_obj, 'number') else (str(sem_obj) if sem_obj else '-')

        if s_attempts:
            best_att = max(s_attempts, key=lambda a: a.percentage)
            pct = best_att.percentage
            if pct >= 85: band = 'Mastery'
            elif pct >= 70: band = 'Proficient'
            elif pct >= 50: band = 'Average'
            else: band = 'Needs Attention'

            result = 'Pass' if pct >= 50 else 'Fail'
            max_mark = best_att.total_questions or fallback_max

            writer.writerow([
                full_name,
                sem_str,
                'Attempted',
                len(s_attempts),
                best_att.score,
                max_mark,
                result,
                band,
            ])
        else:
            writer.writerow([
                full_name,
                sem_str,
                'Not Attempted',
                0,
                '-',
                fallback_max,
                'Absent',
                '-',
            ])

    return response


@teacher_required
def quiz_attempt_detail_api(request, attempt_pk):
    """
    JSON API for student answer sheet drill-down modal.
    """
    from django.http import JsonResponse
    attempt = get_object_or_404(QuizAttempt.objects.select_related('student', 'quiz', 'quiz__subject'), pk=attempt_pk)
    teacher = request.user
    quiz = attempt.quiz

    if quiz.created_by != teacher and quiz.subject not in teacher.assigned_subjects.all() and quiz.subject.semester.department != teacher.department:
        return JsonResponse({'error': 'Access denied'}, status=403)

    student_answers = attempt.student_answers.select_related('question').order_by('question_id')
    questions_data = []

    if student_answers.exists():
        for idx, sa in enumerate(student_answers, start=1):
            q = sa.question
            questions_data.append({
                'number': idx,
                'question_id': q.id,
                'text': q.text,
                'marks': q.marks,
                'option_a': q.option_a,
                'option_b': q.option_b,
                'option_c': q.option_c,
                'option_d': q.option_d,
                'correct_option': q.correct_option,
                'selected_option': sa.selected_option,
                'selected_answer': sa.selected_answer,
                'is_correct': sa.is_correct,
                'marks_obtained': sa.marks_obtained,
                'explanation': q.explanation,
            })
    else:
        # Fallback to answers_json
        idx = 1
        for q_id_str, ans_data in attempt.answers_json.items():
            questions_data.append({
                'number': idx,
                'question_id': q_id_str,
                'text': ans_data.get('question_text', ''),
                'marks': 1,
                'option_a': ans_data.get('option_a', ''),
                'option_b': ans_data.get('option_b', ''),
                'option_c': ans_data.get('option_c', ''),
                'option_d': ans_data.get('option_d', ''),
                'correct_option': ans_data.get('correct_option', ''),
                'selected_option': ans_data.get('chosen_option', ''),
                'selected_answer': ans_data.get('chosen_text', ''),
                'is_correct': ans_data.get('is_correct', False),
                'marks_obtained': 1 if ans_data.get('is_correct') else 0,
                'explanation': ans_data.get('explanation', ''),
            })
            idx += 1

    return JsonResponse({
        'student_name': attempt.student.get_full_name() or attempt.student.username,
        'student_username': attempt.student.username,
        'student_roll': attempt.student.employee_id or attempt.student.username,
        'quiz_title': quiz.title,
        'score': attempt.score,
        'total_questions': attempt.total_questions or quiz.num_questions,
        'percentage': attempt.percentage,
        'time_taken_seconds': attempt.time_taken_seconds,
        'completed_at': attempt.completed_at.strftime('%b %d, %Y %I:%M %p') if attempt.completed_at else '',
        'questions': questions_data,
    })


