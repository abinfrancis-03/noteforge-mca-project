from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from django.core.validators import FileExtensionValidator, MinValueValidator, MaxValueValidator


class Department(models.Model):
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # hod is set via a reverse reference from User.is_hod; helper property below
    hod = models.OneToOneField(
        'User', null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='hod_of_department',
        limit_choices_to={'role': 'teacher'},
    )

    def __str__(self):
        return self.name


class Batch(models.Model):
    department       = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='batches', null=True, blank=True)
    current_semester = models.ForeignKey('Semester', on_delete=models.SET_NULL, null=True, blank=True, related_name='active_batches')
    name             = models.CharField(max_length=60, blank=True)
    start_month      = models.PositiveSmallIntegerField()
    start_year       = models.PositiveIntegerField()
    end_month        = models.PositiveSmallIntegerField()
    end_year         = models.PositiveIntegerField()

    class Meta:
        ordering = ['-start_year', '-start_month', '-id']

    def __str__(self):
        return self.name or f"{self.start_year}–{self.end_year}"

    @property
    def is_ongoing(self):
        now = timezone.now().date()
        return (now.year, now.month) <= (self.end_year, self.end_month)

    def save(self, *args, **kwargs):
        if not self.name:
            self.name = f"{self.start_year}–{self.end_year}"
        if not self.current_semester_id and self.department_id:
            sem1 = Semester.objects.filter(department_id=self.department_id, number=1).first() or Semester.objects.filter(department_id=self.department_id).order_by('number').first()
            if sem1:
                self.current_semester = sem1
        super().save(*args, **kwargs)


class Semester(models.Model):
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='semesters')
    number = models.PositiveIntegerField()

    class Meta:
        unique_together = ('department', 'number')
        ordering = ['number']

    def __str__(self):
        return f"Semester {self.number} — {self.department.name}"


class Subject(models.Model):
    semester     = models.ForeignKey(Semester, on_delete=models.CASCADE, related_name='subjects')
    name         = models.CharField(max_length=150)
    subject_code = models.CharField(max_length=30, unique=True)
    credits      = models.PositiveSmallIntegerField(default=0)

    def __str__(self):
        return f"{self.subject_code} — {self.name}"


class User(AbstractUser):
    ROLE_CHOICES = [
        ('admin',   'Admin'),
        ('teacher', 'Teacher'),
        ('student', 'Student'),
    ]
    DESIGNATION_CHOICES = [
        ('assistant_professor',  'Assistant Professor'),
        ('associate_professor',  'Associate Professor'),
    ]

    role        = models.CharField(max_length=10, choices=ROLE_CHOICES, default='student')
    department  = models.ForeignKey(
        Department, null=True, blank=True,
        on_delete=models.SET_NULL, related_name='users'
    )
    # Students: link to a Semester object
    semester    = models.ForeignKey(
        Semester, null=True, blank=True,
        on_delete=models.SET_NULL, related_name='students'
    )
    # Legacy field kept for backward compat (no longer used in forms)
    semester_number = models.PositiveIntegerField(null=True, blank=True)

    # Batch for students
    batch       = models.ForeignKey(
        Batch, null=True, blank=True,
        on_delete=models.SET_NULL, related_name='students'
    )

    employee_id   = models.CharField(max_length=30, blank=True)
    phone         = models.CharField(max_length=15, blank=True)
    is_first_login = models.BooleanField(default=True)

    # Teacher-specific
    designation       = models.CharField(max_length=30, choices=DESIGNATION_CHOICES, blank=True)
    is_hod            = models.BooleanField(default=False)
    date_of_birth     = models.DateField(null=True, blank=True)
    assigned_subjects = models.ManyToManyField(Subject, blank=True, related_name='teachers')

    def __str__(self):
        return f"{self.username} ({self.role})"


class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('create', 'Created'),
        ('update', 'Updated'),
        ('delete', 'Deleted'),
        ('login',  'Logged In'),
        ('reset',  'Password Reset'),
    ]
    actor        = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    action       = models.CharField(max_length=10, choices=ACTION_CHOICES)
    target_model = models.CharField(max_length=60)
    description  = models.TextField()
    timestamp    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.actor} — {self.action} — {self.target_model}"


class StudyMaterial(models.Model):
    subject       = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='materials')
    title         = models.CharField(max_length=200)
    description   = models.TextField(blank=True)
    file          = models.FileField(upload_to='study_materials/', blank=True, null=True)
    external_link = models.URLField(blank=True)
    uploaded_by   = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_materials')
    created_at    = models.DateTimeField(auto_now_add=True)

    # Material-specific AI Study Summary cache
    ai_summary            = models.TextField(blank=True, null=True)
    ai_summary_json       = models.JSONField(default=dict, blank=True)
    ai_summary_updated_at = models.DateTimeField(null=True, blank=True)

    # Material-specific Visual Mind-Map Image cache
    mindmap_image         = models.ImageField(upload_to='mindmaps/', blank=True, null=True)
    mindmap_updated_at    = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} ({self.subject.subject_code})"


class MaterialDownload(models.Model):
    """Tracks teacher-uploaded materials downloaded or opened by students."""
    student       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='material_downloads')
    material      = models.ForeignKey(StudyMaterial, on_delete=models.CASCADE, related_name='downloads')
    downloaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-downloaded_at']
        unique_together = ('student', 'material')
        verbose_name = 'Material Download'
        verbose_name_plural = 'Material Downloads'

    def __str__(self):
        return f"{self.student.username} downloaded {self.material.title}"


class StudyMaterialChunk(models.Model):
    """
    RAG semantic chunk for teacher-uploaded study materials.
    Stores clean extracted content, source reference (page/slide), and embedding vector.
    """
    material    = models.ForeignKey(StudyMaterial, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.PositiveIntegerField(help_text='Sequential chunk index within material')
    title       = models.CharField(max_length=255, blank=True, default='')
    unit_type   = models.CharField(max_length=50, default='page')  # 'page', 'slide', 'section'
    unit_number = models.PositiveIntegerField(default=1)
    content     = models.TextField(help_text='Clean extracted text content for this chunk')
    token_count = models.PositiveIntegerField(default=0)
    char_count  = models.PositiveIntegerField(default=0)
    embedding   = models.JSONField(default=list, blank=True, help_text='Dense embedding vector')
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['material', 'chunk_index']
        unique_together = ('material', 'chunk_index')

    def __str__(self):
        return f"{self.material.title} — Chunk {self.chunk_index} ({self.unit_type} {self.unit_number})"


class Quiz(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy (Direct Understanding)'),
        ('medium', 'Medium (Concept Connection / Application)'),
        ('hard', 'Hard (Multi-step Reasoning / Analysis)'),
        ('mixed', 'Mixed (Balanced: 25% Easy, 50% Medium, 25% Hard)'),
    ]
    QUESTION_TYPE_CHOICES = [
        ('mcq_single', 'Single Choice MCQ'),
        ('mcq_multi', 'Multiple Response MCQ'),
        ('concept_application', 'Conceptual / Application-based MCQ'),
    ]

    subject         = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='quizzes')
    batch           = models.ForeignKey(Batch, null=True, blank=True, on_delete=models.SET_NULL, related_name='quizzes')
    title           = models.CharField(max_length=200)
    description     = models.TextField(blank=True)
    pdf_file        = models.FileField(
        upload_to='quiz_sources/',
        validators=[FileExtensionValidator(allowed_extensions=['pdf', 'pptx', 'ppt', 'docx', 'doc', 'txt'])],
        blank=True,
        null=True,
        help_text='Source document for MCQ generation'
    )
    num_questions   = models.PositiveIntegerField(
        default=5,
        validators=[MinValueValidator(1), MaxValueValidator(50)],
        help_text='Total questions to configure/generate'
    )
    difficulty      = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='medium')
    question_types  = models.CharField(max_length=50, choices=QUESTION_TYPE_CHOICES, default='mcq_single')
    time_limit_mins = models.PositiveIntegerField(default=15)
    total_marks     = models.PositiveIntegerField(default=10)
    is_published    = models.BooleanField(default=False)
    published_at    = models.DateTimeField(null=True, blank=True, help_text='Timestamp when the quiz was officially published to students')
    # Phase 15: Personalized Practice ⭐ (Adaptive RAG-based Question Generation)
    is_practice      = models.BooleanField(default=False, help_text='Flag indicating a personalized adaptive practice quiz')
    practice_student = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE, related_name='personalized_quizzes', help_text='Student for whom this personalized practice was generated')
    practice_topic   = models.CharField(max_length=200, blank=True, default='', help_text='Targeted weak topic for personalized practice')
    parent_quiz      = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='practice_quizzes', help_text='Source assessment whence the weak topic was diagnosed')
    created_by      = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_quizzes')
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} — {self.subject.subject_code}"


class QuizExtractedPage(models.Model):
    EXTRACTION_METHOD_CHOICES = [
        ('fitz', 'PyMuPDF Text'),
        ('ocr', 'OCR Text'),
        ('empty_scanned', 'Scanned / OCR Pending'),
    ]

    quiz              = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='extracted_pages')
    page_number       = models.PositiveIntegerField(help_text='1-based original PDF page number')
    clean_text        = models.TextField(blank=True, help_text='Cleaned academic content without headers/footers')
    raw_char_count    = models.PositiveIntegerField(default=0)
    clean_char_count  = models.PositiveIntegerField(default=0)
    extraction_method = models.CharField(max_length=20, choices=EXTRACTION_METHOD_CHOICES, default='fitz')
    ocr_needed        = models.BooleanField(default=False, help_text='Flagged when page has insufficient extractable text')
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['quiz', 'page_number']
        unique_together = ('quiz', 'page_number')

    def __str__(self):
        return f"{self.quiz.title} — Page {self.page_number} ({self.clean_char_count} chars)"


class QuizSemanticChunk(models.Model):
    quiz                = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='chunks')
    chunk_index         = models.PositiveIntegerField(help_text='1-based sequential chunk index')
    chunk_id            = models.CharField(max_length=30, help_text='Formatted ID, e.g. CHK-01')
    topic               = models.CharField(max_length=200, help_text='Chapter/Section/Topic boundary label')
    content             = models.TextField(help_text='Semantic chunk text body')
    token_count         = models.PositiveIntegerField(default=0, help_text='Token count (~800-1200 target)')
    char_count          = models.PositiveIntegerField(default=0)
    start_page          = models.PositiveIntegerField(help_text='Source PDF start page')
    end_page            = models.PositiveIntegerField(help_text='Source PDF end page')
    overlap_token_count = models.PositiveIntegerField(default=0, help_text='Context overlap from preceding chunk')
    created_at          = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['quiz', 'chunk_index']
        unique_together = ('quiz', 'chunk_index')

    def __str__(self):
        return f"{self.chunk_id}: {self.topic} (p.{self.start_page}–{self.end_page}, {self.token_count} tokens)"


class Question(models.Model):
    OPTION_CHOICES = [
        ('A', 'Option A'),
        ('B', 'Option B'),
        ('C', 'Option C'),
        ('D', 'Option D'),
    ]
    quiz           = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    text           = models.TextField()
    marks          = models.PositiveIntegerField(default=1)
    option_a       = models.TextField(help_text='Option A text')
    option_b       = models.TextField(help_text='Option B text')
    option_c       = models.TextField(help_text='Option C text')
    option_d       = models.TextField(help_text='Option D text')
    correct_option = models.CharField(max_length=1, choices=OPTION_CHOICES)
    explanation    = models.TextField(blank=True, default='', help_text='Ground-truth explanation from source material')
    topic          = models.CharField(max_length=200, blank=True, default='')
    difficulty     = models.CharField(max_length=20, default='medium')
    blooms_level   = models.CharField(max_length=30, default='Understand')
    question_type  = models.CharField(max_length=50, default='conceptual')
    chunk_id       = models.CharField(max_length=30, blank=True, default='')
    source_page    = models.PositiveIntegerField(default=1)
    source_quote        = models.TextField(blank=True, default='', help_text='Direct citation from retrieved PDF chunk')
    # Phase 9: AI Quality Validation (LLM-as-a-Judge)
    quality_score       = models.PositiveIntegerField(default=0, help_text='Score 0-100 assigned by LLM-as-a-Judge')
    is_approved         = models.BooleanField(default=False, help_text='Approve only questions scoring 85 or above')
    validation_feedback = models.TextField(blank=True, default='', help_text='Detailed academic critique from validator')
    validation_checks   = models.JSONField(default=dict, blank=True, help_text='10-point dimensional checks breakdown')
    validated_at        = models.DateTimeField(null=True, blank=True)
    # Phase 10: Duplicate Detection (Semantic Similarity & Embeddings)
    embedding            = models.JSONField(default=list, blank=True, help_text='Dense embedding vector for semantic similarity')
    is_duplicate         = models.BooleanField(default=False, help_text='Flagged when semantic similarity exceeds threshold')
    duplicate_of         = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='duplicate_instances')
    duplicate_similarity = models.FloatField(default=0.0, help_text='Highest cosine similarity percentage (0-100) with another question')
    duplicate_reason     = models.TextField(blank=True, default='', help_text='AI-assisted duplicate analysis explaining why questions test substantially the same concept')
    # Phase 11: Teacher Review & Governance (Human-in-the-loop AI)
    teacher_approved     = models.BooleanField(default=False, help_text='Explicitly reviewed and approved by the educator')
    teacher_notes        = models.TextField(blank=True, default='', help_text='Educator review notes or remarks')
    reviewed_at          = models.DateTimeField(null=True, blank=True, help_text='Timestamp of teacher review and approval')
    # Phase 12: Relational Quiz Management (Save & Publish)
    APPROVAL_STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]
    source_chunk         = models.ForeignKey('QuizSemanticChunk', null=True, blank=True, on_delete=models.SET_NULL, related_name='questions')
    approval_status      = models.CharField(max_length=20, choices=APPROVAL_STATUS_CHOICES, default='pending', help_text='Status governing student visibility')
    created_at           = models.DateTimeField(auto_now_add=True, null=True)

    @property
    def correct_answer(self):
        """Returns the text of the correct option."""
        mapping = {'A': self.option_a, 'B': self.option_b, 'C': self.option_c, 'D': self.option_d}
        return mapping.get(self.correct_option, self.option_a)

    @property
    def correct_answer_option(self):
        """Phase 12: Returns the relational QuestionOption marked as correct."""
        return self.options.filter(is_correct=True).first()

    def sync_options(self):
        """
        Phase 12 Relational Quiz Management:
        Synchronizes relational QuestionOption rows with Question option_a..d and correct_option.
        """
        options_data = [
            ('A', self.option_a),
            ('B', self.option_b),
            ('C', self.option_c),
            ('D', self.option_d),
        ]
        for letter, opt_text in options_data:
            if opt_text:
                QuestionOption.objects.update_or_create(
                    question=self,
                    option_letter=letter,
                    defaults={
                        'text': opt_text,
                        'is_correct': (self.correct_option == letter),
                    }
                )

    def save(self, *args, **kwargs):
        # Synchronize approval status
        if self.teacher_approved or self.is_approved:
            self.approval_status = 'approved'
        elif self.approval_status == 'approved' and not self.teacher_approved and not self.is_approved:
            self.approval_status = 'pending'

        # Link source_chunk if chunk_id is given and source_chunk is empty
        if self.chunk_id and not self.source_chunk_id and self.quiz_id:
            chunk = self.quiz.chunks.filter(chunk_id=self.chunk_id).first()
            if chunk:
                self.source_chunk = chunk

        super().save(*args, **kwargs)
        # Synchronize relational child options
        self.sync_options()

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f"[{self.difficulty.upper()}] Q: {self.text[:50]}"


class QuestionOption(models.Model):
    """
    Phase 12: Relational quiz management.
    Relational hierarchy: Quiz -> Question -> QuestionOption -> is_correct (Correct Answer)
    """
    OPTION_LETTER_CHOICES = [
        ('A', 'Option A'),
        ('B', 'Option B'),
        ('C', 'Option C'),
        ('D', 'Option D'),
    ]

    question      = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='options')
    option_letter = models.CharField(max_length=1, choices=OPTION_LETTER_CHOICES)
    text          = models.TextField(help_text='Option statement text')
    is_correct    = models.BooleanField(default=False, help_text='True if this option is the designated correct answer')
    created_at    = models.DateTimeField(auto_now_add=True, null=True)

    class Meta:
        ordering = ['option_letter']
        unique_together = ('question', 'option_letter')
        verbose_name = 'Question Option'
        verbose_name_plural = 'Question Options'

    def __str__(self):
        status = " (Correct)" if self.is_correct else ""
        return f"[{self.question.id}-{self.option_letter}] {self.text[:40]}{status}"


class QuizAttempt(models.Model):
    student            = models.ForeignKey(User, on_delete=models.CASCADE, related_name='quiz_attempts')
    quiz               = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='attempts')
    attempt_number     = models.PositiveIntegerField(default=1, help_text='Sequential attempt number for this student and quiz')
    score              = models.PositiveIntegerField(default=0)
    total_questions    = models.PositiveIntegerField(default=0)
    percentage         = models.FloatField(default=0.0)
    time_taken_seconds = models.PositiveIntegerField(default=0)
    answers_json       = models.JSONField(default=dict, blank=True)
    topic_analysis     = models.JSONField(default=dict, blank=True, help_text='Phase 14: Topic-wise learning analytics breakdown, weak topics, and revision recommendations')
    completed_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-completed_at']

    def __str__(self):
        return f"{self.student.username} — {self.quiz.title} (Attempt #{self.attempt_number}: {self.score}/{self.total_questions})"


class StudentAnswer(models.Model):
    """
    Phase 13: Conventional online assessment per-question student response.
    Stores: Student, Quiz, Question, Selected answer, Correct/incorrect, Score, Time, Attempt.
    """
    attempt         = models.ForeignKey(QuizAttempt, on_delete=models.CASCADE, related_name='student_answers')
    question        = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='student_answers')
    selected_option = models.CharField(max_length=1, blank=True, help_text='Selected option letter (A, B, C, D) or blank if skipped')
    selected_answer = models.TextField(blank=True, help_text='Text content of selected option')
    is_correct      = models.BooleanField(default=False, help_text='True if selected answer matches question correct_option')
    marks_obtained  = models.PositiveIntegerField(default=0, help_text='Marks awarded for this question')
    created_at      = models.DateTimeField(auto_now_add=True, null=True)

    class Meta:
        ordering = ['question_id']
        unique_together = ('attempt', 'question')
        verbose_name = 'Student Answer'
        verbose_name_plural = 'Student Answers'

    def __str__(self):
        status = "Correct" if self.is_correct else "Incorrect"
        return f"{self.attempt.student.username} - Q#{self.question_id}: {self.selected_option} ({status})"


class SavedSummary(models.Model):
    SUMMARY_TYPES = [
        ('document', 'Document Summary'),
        ('youtube', 'YouTube Lecture Summary'),
        ('mindmap', 'Mind Map'),
    ]
    student         = models.ForeignKey(User, on_delete=models.CASCADE, related_name='saved_summaries')
    source_material = models.ForeignKey(StudyMaterial, null=True, blank=True, on_delete=models.SET_NULL, related_name='summaries')
    title           = models.CharField(max_length=200)
    summary_text    = models.TextField()
    summary_type    = models.CharField(max_length=20, choices=SUMMARY_TYPES, default='document')
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} — {self.student.username}"


class QuizAIAnalysis(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('analyzing', 'Analyzing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    quiz                = models.OneToOneField(Quiz, on_delete=models.CASCADE, related_name='ai_analysis')
    status              = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    structured_json     = models.JSONField(default=dict, blank=True)
    model_name          = models.CharField(max_length=100, default='gemini-2.5-flash')
    prompt_tokens       = models.PositiveIntegerField(default=0)
    completion_tokens   = models.PositiveIntegerField(default=0)
    total_tokens        = models.PositiveIntegerField(default=0)
    total_topics        = models.PositiveIntegerField(default=0)
    total_subtopics     = models.PositiveIntegerField(default=0)
    total_definitions   = models.PositiveIntegerField(default=0)
    total_concepts      = models.PositiveIntegerField(default=0)
    total_processes     = models.PositiveIntegerField(default=0)
    total_relationships = models.PositiveIntegerField(default=0)
    total_comparisons   = models.PositiveIntegerField(default=0)
    total_applications  = models.PositiveIntegerField(default=0)
    total_formulas      = models.PositiveIntegerField(default=0)
    error_message       = models.TextField(blank=True, null=True)
    created_at          = models.DateTimeField(auto_now_add=True)
    updated_at          = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Quiz AI Analysis'
        verbose_name_plural = 'Quiz AI Analyses'

    def __str__(self):
        return f"AI Analysis for {self.quiz.title} ({self.status})"


class QuizRankedConcept(models.Model):
    IMPORTANCE_CHOICES = [
        ('HIGH', 'High Priority'),
        ('MEDIUM', 'Medium Priority'),
        ('LOW', 'Low Priority'),
    ]

    quiz              = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='ranked_concepts')
    concept           = models.CharField(max_length=255)
    category          = models.CharField(max_length=100, default='Core Concept')
    importance        = models.CharField(max_length=10, choices=IMPORTANCE_CHOICES, default='MEDIUM')
    importance_score  = models.PositiveIntegerField(default=50, help_text='0-100 numerical score')
    reason            = models.TextField(help_text='Educational prioritization rationale based only on source text')
    chunk_id          = models.CharField(max_length=30, blank=True, default='')
    start_page        = models.PositiveIntegerField(default=1)
    end_page          = models.PositiveIntegerField(default=1)
    criteria_metadata = models.JSONField(default=dict, blank=True)
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-importance_score', 'concept']
        verbose_name = 'Quiz Ranked Concept'
        verbose_name_plural = 'Quiz Ranked Concepts'

    def __str__(self):
        return f"[{self.importance} - {self.importance_score}] {self.concept} ({self.quiz.title})"




