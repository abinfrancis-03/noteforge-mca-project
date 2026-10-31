from django.urls import path
from . import views

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='admin_dashboard'),

    # Departments
    path('departments/',              views.department_list,   name='department_list'),
    path('departments/add/',          views.department_add,    name='department_add'),
    path('departments/<int:pk>/edit/', views.department_edit,  name='department_edit'),
    path('departments/<int:pk>/delete/', views.department_delete, name='department_delete'),

    # Semesters
    path('semesters/',                views.semester_list,     name='semester_list'),
    path('semesters/add/',            views.semester_add,      name='semester_add'),
    path('semesters/<int:pk>/delete/', views.semester_delete,  name='semester_delete'),

    # Subjects
    path('subjects/',                 views.subject_list,      name='subject_list'),
    path('subjects/add/',             views.subject_add,       name='subject_add'),
    path('subjects/<int:pk>/edit/',   views.subject_edit,      name='subject_edit'),
    path('subjects/<int:pk>/delete/', views.subject_delete,    name='subject_delete'),

    # Teachers
    path('teachers/',                 views.teacher_list,      name='teacher_list'),
    path('teachers/add/',             views.teacher_add,       name='teacher_add'),
    path('teachers/<int:pk>/',         views.teacher_detail,    name='teacher_detail'),
    path('teachers/<int:pk>/edit/',   views.teacher_edit,      name='teacher_edit'),
    path('teachers/<int:pk>/delete/', views.teacher_delete,    name='teacher_delete'),
    path('teachers/<int:pk>/reset-password/', views.teacher_reset_password, name='teacher_reset_password'),

    # Batches
    path('batches/',                  views.batch_list,        name='batch_list'),
    path('batches/add/',              views.batch_add,         name='batch_add'),
    path('batches/<int:pk>/edit/',    views.batch_edit,        name='batch_edit'),
    path('batches/<int:pk>/delete/',  views.batch_delete,      name='batch_delete'),

    # Students
    path('students/',                 views.student_list,      name='student_list'),
    path('students/add/',             views.student_add,       name='student_add'),
    path('students/promote/',         views.student_promote,   name='student_promote'),
    path('students/<int:pk>/edit/',   views.student_edit,      name='student_edit'),
    path('students/<int:pk>/delete/', views.student_delete,    name='student_delete'),
    path('students/<int:pk>/reset-password/', views.student_reset_password, name='student_reset_password'),

    # API helpers (AJAX)
    path('api/semesters/',            views.api_semesters,     name='api_semesters'),
    path('api/batches/',              views.api_batches,       name='api_batches'),
    path('api/dept-semesters/',       views.api_departments_semesters, name='api_dept_semesters'),
    path('api/department-subjects/',  views.api_department_subjects,  name='api_department_subjects'),

    # Audit Log
    path('audit-log/', views.audit_log, name='audit_log'),
]
