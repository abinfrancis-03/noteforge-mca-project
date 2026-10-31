from django import forms
from django.core.exceptions import ValidationError
from accounts.models import Quiz, Subject, Semester, Batch, Question


class QuizConfigurationForm(forms.ModelForm):
    semester = forms.ModelChoiceField(
        queryset=Semester.objects.none(),
        required=True,
        empty_label="— Select Semester —",
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'id_semester',
            'onchange': 'loadSubjectsForSem()',
        })
    )

    class Meta:
        model = Quiz
        fields = [
            'title',
            'subject',
            'batch',
            'pdf_file',
            'num_questions',
            'time_limit_mins',
            'description',
        ]
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Mid-Term Assessment — Operating Systems',
                'required': True,
            }),
            'subject': forms.Select(attrs={
                'class': 'form-select',
                'id': 'id_subject',
                'required': True,
            }),
            'batch': forms.Select(attrs={
                'class': 'form-select',
            }),
            'pdf_file': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.pptx,.ppt,.docx,.doc,.txt,application/pdf,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,text/plain',
                'required': True,
            }),
            'num_questions': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 50,
                'value': 5,
                'required': True,
            }),
            'time_limit_mins': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 180,
                'value': 15,
                'required': True,
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'Optional guidelines or syllabus notes for this test...',
            }),
        }

    def save(self, commit=True):
        quiz = super().save(commit=False)
        if not quiz.difficulty or quiz.difficulty == 'medium':
            quiz.difficulty = 'mixed'
        if not quiz.question_types:
            quiz.question_types = 'mcq_single'
        if commit:
            quiz.save()
        return quiz

    def __init__(self, *args, teacher=None, locked_batch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher

        if teacher:
            assigned_subjects = teacher.assigned_subjects.all()
            self.fields['subject'].queryset = assigned_subjects.order_by('subject_code')

            # Populate semesters that have subjects assigned to this teacher
            sem_ids = assigned_subjects.values_list('semester_id', flat=True).distinct()
            self.fields['semester'].queryset = Semester.objects.filter(id__in=sem_ids).order_by('number')

            if teacher.department:
                self.fields['batch'].queryset = Batch.objects.filter(department=teacher.department)
            else:
                self.fields['batch'].queryset = Batch.objects.all()

        if locked_batch:
            # Batch is locked from the select-batch step — make it required & pre-set
            self.fields['batch'].required = True
            self.fields['batch'].initial = locked_batch
            self.fields['batch'].widget.attrs['disabled'] = True
            # Pre-set semester to this batch's current semester
            if locked_batch.current_semester:
                self.fields['semester'].initial = locked_batch.current_semester
        else:
            self.fields['batch'].required = False
            self.fields['batch'].empty_label = "— All Batches (Universal) —"

        self.fields['subject'].empty_label = "— Select Subject —"

        # Pre-select semester if instance exists or form bound
        if self.instance and self.instance.pk and self.instance.subject:
            self.fields['semester'].initial = self.instance.subject.semester_id

    def clean_pdf_file(self):
        pdf = self.cleaned_data.get('pdf_file')
        if not pdf:
            raise ValidationError("A source study material (PDF, PPTX, DOCX) is required for MCQ test configuration.")

        # 1. File extension validation
        ext = pdf.name.rsplit('.', 1)[-1].lower() if '.' in pdf.name else ''
        allowed_exts = ('pdf', 'pptx', 'ppt', 'docx', 'doc', 'txt')
        if ext not in allowed_exts:
            raise ValidationError("Invalid file format. Accepted formats: PDF, PPTX, PPT, DOCX, DOC, TXT.")

        # 2. File size limit: 35 MB max
        max_bytes = 35 * 1024 * 1024
        if pdf.size > max_bytes:
            raise ValidationError("The uploaded file is too large. Maximum allowed size is 35 MB.")

        return pdf

    def clean(self):
        cleaned_data = super().clean()
        subject = cleaned_data.get('subject')
        semester = cleaned_data.get('semester')

        if subject and semester and subject.semester_id != semester.id:
            raise ValidationError({
                'subject': "The selected subject does not belong to the selected semester."
            })

        return cleaned_data


class QuestionReviewForm(forms.ModelForm):
    """
    Phase 11: Teacher Review & Human-in-the-Loop Editing Form.
    Allows educator to edit question text, 4 options, correct answer, explanation,
    topic, cognitive difficulty tier, and Bloom's taxonomy level.
    """
    class Meta:
        model = Question
        fields = [
            'text',
            'option_a',
            'option_b',
            'option_c',
            'option_d',
            'correct_option',
            'explanation',
            'topic',
            'difficulty',
            'blooms_level',
        ]
        widgets = {
            'text': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'required': True}),
            'option_a': forms.TextInput(attrs={'class': 'form-control', 'required': True}),
            'option_b': forms.TextInput(attrs={'class': 'form-control', 'required': True}),
            'option_c': forms.TextInput(attrs={'class': 'form-control', 'required': True}),
            'option_d': forms.TextInput(attrs={'class': 'form-control', 'required': True}),
            'correct_option': forms.Select(attrs={'class': 'form-select', 'required': True}),
            'explanation': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'required': True}),
            'topic': forms.TextInput(attrs={'class': 'form-control', 'required': True}),
            'difficulty': forms.Select(
                choices=[
                    ('easy', 'Easy (Direct Understanding)'),
                    ('medium', 'Medium (Concept Connection)'),
                    ('hard', 'Hard (Multi-step Reasoning)'),
                ],
                attrs={'class': 'form-select', 'required': True}
            ),
            'blooms_level': forms.Select(
                choices=[
                    ('Remember', 'Remember'),
                    ('Understand', 'Understand'),
                    ('Apply', 'Apply'),
                    ('Analyse', 'Analyse'),
                ],
                attrs={'class': 'form-select', 'required': True}
            ),
        }

