from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from accounts.models import AuditLog


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            AuditLog.objects.create(
                actor=user, action='login',
                target_model='User',
                description=f'{user.username} logged in.'
            )
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


def home_redirect(request):
    if not request.user.is_authenticated:
        return redirect('login')
    role = getattr(request.user, 'role', None)
    if role == 'admin':
        return redirect('admin_dashboard')
    elif role == 'teacher':
        if request.user.is_first_login:
            return redirect('teacher_change_password')
        return redirect('teacher_dashboard')
    elif role == 'student':
        if request.user.is_first_login:
            return redirect('student_change_password')
        return redirect('student_dashboard')
    return redirect('login')

