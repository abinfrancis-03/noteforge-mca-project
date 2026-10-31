from django.urls import path
from student_panel import views

urlpatterns = [
    path('dashboard/', views.dashboard, name='student_dashboard'),
    path('change-password/', views.change_password, name='student_change_password'),
    path('materials/', views.material_list, name='student_material_list'),
    path('materials/<int:pk>/download/', views.download_material, name='student_download_material'),
    path('quizzes/', views.quiz_list, name='student_quiz_list'),
    path('quizzes/<int:pk>/take/', views.quiz_take, name='student_quiz_take'),
    path('quizzes/<int:pk>/submit/', views.quiz_submit, name='student_quiz_submit'),
    path('quizzes/<int:pk>/result/<int:attempt_id>/', views.quiz_result, name='student_quiz_result'),
    path('quizzes/<int:pk>/practice/', views.generate_personalized_practice, name='student_generate_practice'),
    path('analytics/', views.student_performance_analytics, name='student_performance_analytics'),

    # Advanced AI Learning Suite
    path('tools/ai-summary/', views.ai_summary_hub, name='student_ai_summary_hub'),
    path('tools/mindmap/', views.mindmap_hub, name='student_mindmap_hub'),
    path('tools/chatbot/', views.chatbot_hub, name='student_chatbot_hub'),
    path('materials/<int:pk>/summarize/', views.document_summarize, name='student_document_summarize'),
    path('library/', views.my_library, name='student_my_library'),
    path('library/<int:pk>/delete/', views.delete_saved_summary, name='student_delete_saved_summary'),
    path('tools/youtube-summarizer/', views.youtube_summarizer, name='student_youtube_summarizer'),
    path('subjects/<int:pk>/chatbot/', views.subject_chatbot, name='student_subject_chatbot'),
    path('api/chatbot/', views.api_chatbot_query, name='student_api_chatbot'),
    path('tools/save-gemini-key/', views.save_gemini_key, name='student_save_gemini_key'),
    path('materials/<int:pk>/mindmap/', views.mindmap_generator, name='student_mindmap_generator'),
]

