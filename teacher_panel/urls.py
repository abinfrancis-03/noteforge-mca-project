from django.urls import path
from teacher_panel import views

urlpatterns = [
    path('dashboard/', views.dashboard, name='teacher_dashboard'),
    path('change-password/', views.change_password, name='teacher_change_password'),
    path('materials/', views.material_list, name='teacher_material_list'),
    path('materials/add/', views.material_add, name='teacher_material_add'),
    path('materials/<int:pk>/edit/', views.material_edit, name='teacher_material_edit'),
    path('materials/<int:pk>/delete/', views.material_delete, name='teacher_material_delete'),
    
    # AI Quiz Generator from PDF
    path('materials/<int:pk>/generate-quiz/', views.ai_quiz_generate, name='teacher_ai_quiz_generate'),

    # Student Directory (Duplicate-Resistant)
    path('students/', views.student_directory, name='teacher_student_directory'),

    # Quizzes
    path('quizzes/', views.quiz_list, name='teacher_quiz_list'),
    path('quizzes/create/select-batch/', views.quiz_select_batch, name='teacher_quiz_select_batch'),
    path('quizzes/create/', views.quiz_create, name='teacher_quiz_create'),
    path('quizzes/<int:pk>/extracted-text/', views.quiz_extracted_text, name='teacher_quiz_extracted_text'),
    path('quizzes/<int:pk>/chunks/', views.quiz_chunks, name='teacher_quiz_chunks'),
    path('quizzes/<int:pk>/analysis/', views.quiz_analysis, name='teacher_quiz_analysis'),
    path('quizzes/<int:pk>/ranked-concepts/', views.quiz_ranked_concepts, name='teacher_quiz_ranked_concepts'),
    path('quizzes/<int:pk>/questions/', views.quiz_questions, name='teacher_quiz_questions'),
    path('quizzes/<int:pk>/delete/', views.quiz_delete, name='teacher_quiz_delete'),
    
    # Phase 11 & Phase 12: Teacher Review & Relational Quiz Management (Save & Publish)
    path('quizzes/<int:quiz_pk>/questions/<int:question_pk>/edit/', views.question_edit, name='teacher_question_edit'),
    path('quizzes/<int:quiz_pk>/questions/<int:question_pk>/regenerate/', views.question_regenerate, name='teacher_question_regenerate'),
    path('quizzes/<int:quiz_pk>/questions/<int:question_pk>/approve/', views.question_approve, name='teacher_question_approve'),
    path('quizzes/<int:quiz_pk>/questions/<int:question_pk>/delete/', views.question_delete, name='teacher_question_delete'),
    path('quizzes/<int:quiz_pk>/approve-all/', views.quiz_approve_all, name='teacher_quiz_approve_all'),
    path('quizzes/<int:quiz_pk>/save-approved/', views.quiz_save_approved, name='teacher_quiz_save_approved'),
    path('quizzes/<int:quiz_pk>/publish/', views.quiz_publish, name='teacher_quiz_publish'),
    
    # MCQ Student Analytics & Results
    path('quizzes/analytics/', views.quiz_analytics_dashboard, name='teacher_quiz_analytics'),
    path('quizzes/<int:quiz_pk>/analytics/', views.quiz_test_analytics, name='teacher_quiz_test_analytics'),
    path('quizzes/<int:quiz_pk>/analytics/export/', views.quiz_analytics_export, name='teacher_quiz_analytics_export'),
    path('quizzes/attempts/<int:attempt_pk>/detail/', views.quiz_attempt_detail_api, name='teacher_quiz_attempt_detail_api'),
    
    path('api/subjects/', views.api_subjects, name='teacher_api_subjects'),
    path('api/subject-materials/', views.api_subject_materials, name='teacher_api_subject_materials'),
    path('api/batch-info/', views.api_batch_info, name='teacher_api_batch_info'),
    path('hod-overview/', views.hod_overview, name='teacher_hod_overview'),
]
