from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Department, Semester, Subject, AuditLog

admin.site.register(User, UserAdmin)
admin.site.register(Department)
admin.site.register(Semester)
admin.site.register(Subject)
admin.site.register(AuditLog)
