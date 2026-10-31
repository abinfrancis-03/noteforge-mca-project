import json
import random
import string

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from django.db.models import Count, Q
from django.http import JsonResponse

from django.utils import timezone
from accounts.models import Department, Batch, Semester, Subject, User, AuditLog, StudyMaterial



# ─── helpers ──────────────────────────────────────────────────────────────────

def admin_required(view_func):
    """Decorator: login + admin role required."""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        if request.user.role != 'admin':
            messages.error(request, 'Access denied.')
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return wrapper


def redirect_back(request, fallback):
    """Redirect to next parameter or HTTP referrer or fallback URL."""
    next_url = request.POST.get('next') or request.GET.get('next') or request.META.get('HTTP_REFERER')
    if next_url:
        return redirect(next_url)
    return redirect(fallback)


def log(actor, action, model, desc):
    AuditLog.objects.create(actor=actor, action=action, target_model=model, description=desc)


def dob_password(dob):
    """Generate password from date_of_birth as DDMMYYYY string."""
    if dob:
        return dob.strftime('%d%m%Y')
    # fallback random
    chars = string.ascii_letters + string.digits + '!@#$'
    return ''.join(random.choices(chars, k=10))


# ─── dashboard ────────────────────────────────────────────────────────────────

@admin_required
def dashboard(request):
    now = timezone.now().date()
    ongoing_batches = Batch.objects.filter(
        Q(end_year__gt=now.year) | Q(end_year=now.year, end_month__gte=now.month)
    )

    # Storage space metric calculation
    total_storage_bytes = 0
    materials = StudyMaterial.objects.all()
    for m in materials:
        if m.file:
            try:
                total_storage_bytes += m.file.size
            except Exception:
                pass

    total_storage_mb = round(total_storage_bytes / (1024 * 1024), 2)
    total_storage_gb = round(total_storage_bytes / (1024 * 1024 * 1024), 3)

    total_teachers = User.objects.filter(role='teacher').count()
    total_ongoing_students = User.objects.filter(role='student', batch__in=ongoing_batches).count()

    pending_teacher_logins = User.objects.filter(role='teacher', is_first_login=True).count()
    pending_student_logins = User.objects.filter(role='student', batch__in=ongoing_batches, is_first_login=True).count()

    # User account health counts
    active_teachers = max(0, total_teachers - pending_teacher_logins)
    active_students = max(0, total_ongoing_students - pending_student_logins)
    total_active_users = active_teachers + active_students
    total_pending_users = pending_teacher_logins + pending_student_logins
    total_users_count = total_active_users + total_pending_users
    active_users_pct = round((total_active_users / total_users_count * 100), 1) if total_users_count > 0 else 100

    # Storage gauge
    storage_quota_mb = 1000  # 1GB quota benchmark
    storage_pct = min(100, round((total_storage_mb / storage_quota_mb) * 100, 1))

    # Departments with student share percentage
    departments = list(Department.objects.annotate(
        teacher_count=Count('users', filter=Q(users__role='teacher')),
        student_count=Count('users', filter=Q(users__role='student', users__batch__in=ongoing_batches)),
    ).order_by('name'))

    for d in departments:
        d.student_pct = round((d.student_count / total_ongoing_students * 100), 1) if total_ongoing_students > 0 else 0

    # Chart datasets (JSON serialized)
    dept_chart_labels = json.dumps([d.code for d in departments])
    dept_chart_students = json.dumps([d.student_count for d in departments])
    dept_chart_teachers = json.dumps([d.teacher_count for d in departments])

    ongoing_batches_count = ongoing_batches.count()
    total_batches = Batch.objects.count()
    completed_batches_count = max(0, total_batches - ongoing_batches_count)

    # Table view: top 6 previously updated departments (add students, add batch, add teacher, or dept update)
    dept_map = {d.id: d for d in departments}
    recent_dept_ids = []
    
    for alog in AuditLog.objects.exclude(action='login').order_by('-timestamp'):
        did = None
        desc = alog.description
        if alog.target_model == 'Department':
            for d in departments:
                if d.name in desc or f'({d.code})' in desc:
                    did = d.id
                    break
        elif alog.target_model == 'User':
            for d in departments:
                if f'in {d.name}' in desc or f'({d.code})' in desc:
                    did = d.id
                    break
        elif alog.target_model == 'Batch':
            for d in departments:
                if f'for {d.name}' in desc or f'({d.code})' in desc:
                    did = d.id
                    break
        if did and did in dept_map and did not in recent_dept_ids:
            recent_dept_ids.append(did)
            if len(recent_dept_ids) == 6:
                break

    # Fallback to remaining departments by newest ID if fewer than 6 found
    if len(recent_dept_ids) < 6:
        for d in sorted(departments, key=lambda x: x.id, reverse=True):
            if d.id not in recent_dept_ids:
                recent_dept_ids.append(d.id)
                if len(recent_dept_ids) == 6:
                    break

    updated_departments = [dept_map[did] for did in recent_dept_ids if did in dept_map]

    ctx = {
        'total_departments':       Department.objects.count(),
        'total_teachers':          total_teachers,
        'total_batches':           total_batches,
        'ongoing_batches_count':   ongoing_batches_count,
        'completed_batches_count': completed_batches_count,
        'total_ongoing_students':  total_ongoing_students,
        'total_materials':         materials.count(),
        'total_storage_mb':        total_storage_mb,
        'total_storage_gb':        total_storage_gb,
        'storage_pct':             storage_pct,
        'pending_teacher_logins':  pending_teacher_logins,
        'pending_student_logins':  pending_student_logins,
        'total_active_users':      total_active_users,
        'total_pending_users':     total_pending_users,
        'active_users_pct':        active_users_pct,
        'recent_logs':             AuditLog.objects.select_related('actor')[:5],
        'departments':             departments,
        'updated_departments':     updated_departments,
        'dept_chart_labels':       dept_chart_labels,
        'dept_chart_students':     dept_chart_students,
        'dept_chart_teachers':     dept_chart_teachers,
    }
    return render(request, 'admin_panel/dashboard.html', ctx)



# ─── departments ──────────────────────────────────────────────────────────────

@admin_required
def department_list(request):
    now = timezone.now().date()
    ongoing_batches = Batch.objects.filter(
        Q(end_year__gt=now.year) | Q(end_year=now.year, end_month__gte=now.month)
    )
    departments = Department.objects.select_related('hod').annotate(
        sem_count=Count('semesters', distinct=True),
        teacher_count=Count('users', filter=Q(users__role='teacher'), distinct=True),
        student_count=Count('users', filter=Q(users__role='student', users__batch__in=ongoing_batches), distinct=True),
    )
    return render(request, 'admin_panel/department_list.html', {'departments': departments})


@admin_required
def department_add(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        code = request.POST.get('code', '').strip().upper()
        if not name or not code:
            messages.error(request, 'Name and code are required.')
        elif Department.objects.filter(Q(name__iexact=name) | Q(code__iexact=code)).exists():
            messages.error(request, 'Department name or code already exists.')
        else:
            dept = Department.objects.create(name=name, code=code)
            log(request.user, 'create', 'Department', f'Created department: {dept.name} ({dept.code})')
            messages.success(request, f'Department "{dept.name}" created.')
            return redirect_back(request, 'department_list')
    return render(request, 'admin_panel/department_form.html', {'action': 'Add'})


@admin_required
def department_edit(request, pk):
    dept = get_object_or_404(Department, pk=pk)
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        code = request.POST.get('code', '').strip().upper()
        if not name or not code:
            messages.error(request, 'Name and code are required.')
        elif Department.objects.filter(Q(name__iexact=name) | Q(code__iexact=code)).exclude(pk=pk).exists():
            messages.error(request, 'Another department with that name or code exists.')
        else:
            dept.name = name
            dept.code = code
            dept.save()
            log(request.user, 'update', 'Department', f'Updated department: {dept.name} ({dept.code})')
            messages.success(request, f'Department "{dept.name}" updated.')
            return redirect_back(request, 'department_list')
    return render(request, 'admin_panel/department_form.html', {'action': 'Edit', 'dept': dept})


@admin_required
def department_delete(request, pk):
    dept = get_object_or_404(Department, pk=pk)
    if request.method == 'POST':
        name = dept.name
        dept.delete()
        log(request.user, 'delete', 'Department', f'Deleted department: {name}')
        messages.success(request, f'Department "{name}" deleted.')
        return redirect_back(request, 'department_list')
    return render(request, 'admin_panel/confirm_delete.html', {
        'object_name': dept.name, 'cancel_url': 'department_list'
    })


# ─── semesters ────────────────────────────────────────────────────────────────

@admin_required
def semester_list(request):
    dept_id = request.GET.get('dept')
    departments = Department.objects.all()
    selected_dept = None
    semesters = None

    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
            semesters = Semester.objects.filter(department=selected_dept).annotate(
                subject_count=Count('subjects')
            ).order_by('number')
        except Department.DoesNotExist:
            pass

    return render(request, 'admin_panel/semester_list.html', {
        'semesters': semesters,
        'departments': departments,
        'selected_dept': selected_dept,
    })


@admin_required
def semester_add(request):
    dept_id = request.GET.get('dept') or request.POST.get('department')
    if request.method == 'POST':
        total = request.POST.get('total_semesters', '').strip()
        dept_id = request.POST.get('department')
        if not dept_id or not total:
            messages.error(request, 'Department and total semester count are required.')
        else:
            try:
                total = int(total)
                if total < 1 or total > 20:
                    raise ValueError
            except ValueError:
                messages.error(request, 'Total semesters must be a number between 1 and 20.')
            else:
                dept = get_object_or_404(Department, pk=dept_id)
                existing = set(Semester.objects.filter(department=dept).values_list('number', flat=True))
                created = []
                for n in range(1, total + 1):
                    if n not in existing:
                        Semester.objects.create(department=dept, number=n)
                        created.append(n)
                if created:
                    log(request.user, 'create', 'Semester',
                        f'Created semesters {created} in {dept.name}')
                    messages.success(request, f'Created {len(created)} semester(s) in {dept.name}.')
                else:
                    messages.info(request, f'All {total} semesters already exist for {dept.name}.')
                return redirect_back(request, f'/admin-panel/semesters/?dept={dept.pk}')
    selected_dept = None
    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
        except Department.DoesNotExist:
            pass
    return render(request, 'admin_panel/semester_form.html', {
        'action': 'Add',
        'departments': Department.objects.all(),
        'selected_dept': selected_dept,
    })


@admin_required
def semester_delete(request, pk):
    sem = get_object_or_404(Semester, pk=pk)
    dept_pk = sem.department_id
    if request.method == 'POST':
        label = str(sem)
        sem.delete()
        log(request.user, 'delete', 'Semester', f'Deleted {label}')
        messages.success(request, f'{label} deleted.')
        return redirect_back(request, f'/admin-panel/semesters/?dept={dept_pk}')
    return render(request, 'admin_panel/confirm_delete.html', {
        'object_name': str(sem), 'cancel_url': 'semester_list'
    })


# ─── AJAX helpers ─────────────────────────────────────────────────────────────

@admin_required
def api_semesters(request):
    """Return semesters for a department as JSON."""
    dept_id = request.GET.get('dept')
    data = []
    if dept_id:
        sems = Semester.objects.filter(department_id=dept_id).order_by('number')
        data = [{'id': s.pk, 'label': f'Semester {s.number}'} for s in sems]
    return JsonResponse({'semesters': data})


@admin_required
def api_batches(request):
    """Return batches for a department as JSON."""
    dept_id = request.GET.get('dept')
    data = []
    if dept_id:
        batches = Batch.objects.filter(department_id=dept_id).select_related('current_semester').order_by('-start_year', '-start_month', '-id')
        data = [{
            'id': b.pk,
            'label': str(b),
            'current_semester_id': b.current_semester_id,
            'current_semester_name': f'Semester {b.current_semester.number}' if b.current_semester else None
        } for b in batches]
    return JsonResponse({'batches': data})


@admin_required
def api_departments_semesters(request):
    """Return all departments with their semesters."""
    depts = Department.objects.prefetch_related('semesters').all()
    data = {}
    for d in depts:
        data[str(d.pk)] = [{'id': s.pk, 'label': f'Semester {s.number}'} for s in d.semesters.all()]
    return JsonResponse(data)


# ─── subjects ─────────────────────────────────────────────────────────────────

@admin_required
def subject_list(request):
    dept_id = request.GET.get('dept')
    sem_id  = request.GET.get('sem')
    departments = Department.objects.prefetch_related('semesters').all()
    selected_dept = None
    selected_sem  = None
    semesters = []
    subjects  = None


    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
            semesters = Semester.objects.filter(department=selected_dept).order_by('number')
        except (Department.DoesNotExist, ValueError):
            pass

    if sem_id:
        try:
            selected_sem = Semester.objects.get(pk=sem_id)
            if not selected_dept:
                selected_dept = selected_sem.department
                semesters = Semester.objects.filter(department=selected_dept).order_by('number')
            elif selected_sem.department_id != selected_dept.pk:
                selected_sem = None

            if selected_sem:
                subjects = Subject.objects.filter(semester=selected_sem).select_related('semester__department')
        except (Semester.DoesNotExist, ValueError):
            pass

    return render(request, 'admin_panel/subject_list.html', {
        'subjects': subjects,
        'departments': departments,
        'semesters': semesters,
        'selected_dept': selected_dept,
        'selected_sem': selected_sem,
    })



@admin_required
def subject_add(request):
    dept_id = request.GET.get('dept') or request.POST.get('dept_id')
    sem_id  = request.GET.get('sem')  or request.POST.get('semester')

    if request.method == 'POST':
        sem_id  = request.POST.get('semester')
        name    = request.POST.get('name', '').strip()
        code    = request.POST.get('subject_code', '').strip().upper()
        credits = request.POST.get('credits', '0').strip()
        dept_id = request.POST.get('dept_id')

        if not sem_id or not name or not code:
            messages.error(request, 'All fields are required.')
        elif Subject.objects.filter(subject_code__iexact=code).exists():
            messages.error(request, 'Subject code already exists.')
        else:
            try:
                credits_val = int(credits) if credits else 0
            except ValueError:
                credits_val = 0
            sem = get_object_or_404(Semester, pk=sem_id)
            sub = Subject.objects.create(semester=sem, name=name, subject_code=code, credits=credits_val)
            log(request.user, 'create', 'Subject', f'Created subject {sub.subject_code} — {sub.name}')
            messages.success(request, f'Subject "{sub.name}" added.')
            return redirect_back(request, f'/admin-panel/subjects/?dept={sem.department_id}&sem={sem.pk}')

    selected_dept = None
    selected_sem  = None
    semesters = []
    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
            semesters = Semester.objects.filter(department=selected_dept).order_by('number')
        except Department.DoesNotExist:
            pass
    if sem_id:
        try:
            selected_sem = Semester.objects.get(pk=sem_id)
        except Semester.DoesNotExist:
            pass

    return render(request, 'admin_panel/subject_form.html', {
        'action': 'Add',
        'departments': Department.objects.all(),
        'semesters': semesters,
        'selected_dept': selected_dept,
        'selected_sem': selected_sem,
    })


@admin_required
def subject_edit(request, pk):
    sub = get_object_or_404(Subject, pk=pk)
    if request.method == 'POST':
        sem_id  = request.POST.get('semester')
        name    = request.POST.get('name', '').strip()
        code    = request.POST.get('subject_code', '').strip().upper()
        credits = request.POST.get('credits', '0').strip()
        if Subject.objects.filter(subject_code__iexact=code).exclude(pk=pk).exists():
            messages.error(request, 'Subject code already used by another subject.')
        else:
            try:
                credits_val = int(credits) if credits else 0
            except ValueError:
                credits_val = 0
            sub.semester = get_object_or_404(Semester, pk=sem_id)
            sub.name = name
            sub.subject_code = code
            sub.credits = credits_val
            sub.save()
            log(request.user, 'update', 'Subject', f'Updated subject {sub.subject_code} — {sub.name}')
            messages.success(request, f'Subject "{sub.name}" updated.')
            return redirect_back(request, f'/admin-panel/subjects/?dept={sub.semester.department_id}&sem={sub.semester_id}')
    dept = sub.semester.department
    semesters = Semester.objects.filter(department=dept).order_by('number')
    return render(request, 'admin_panel/subject_form.html', {
        'action': 'Edit', 'sub': sub,
        'departments': Department.objects.all(),
        'semesters': semesters,
        'selected_dept': dept,
        'selected_sem': sub.semester,
    })


@admin_required
def subject_delete(request, pk):
    sub = get_object_or_404(Subject, pk=pk)
    if request.method == 'POST':
        name = sub.name
        sem  = sub.semester
        sub.delete()
        log(request.user, 'delete', 'Subject', f'Deleted subject: {name}')
        messages.success(request, f'Subject "{name}" deleted.')
        return redirect_back(request, f'/admin-panel/subjects/?dept={sem.department_id}&sem={sem.pk}')
    return render(request, 'admin_panel/confirm_delete.html', {
        'object_name': sub.name, 'cancel_url': 'subject_list'
    })


# ─── teachers ─────────────────────────────────────────────────────────────────

@admin_required
def teacher_list(request):
    dept_id = request.GET.get('dept')
    departments = Department.objects.all()
    selected_dept = None
    teachers = None

    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
            teachers = User.objects.filter(role='teacher', department=selected_dept).select_related('department').prefetch_related('assigned_subjects', 'assigned_subjects__semester')
        except Department.DoesNotExist:
            pass

    return render(request, 'admin_panel/teacher_list.html', {
        'teachers': teachers,
        'departments': departments,
        'selected_dept': selected_dept,
    })


def _get_department_semesters(dept):
    if not dept:
        return []
    return Semester.objects.filter(department=dept).prefetch_related('subjects').order_by('number')


@admin_required
def teacher_add(request):
    dept_id = request.GET.get('dept') or request.POST.get('department')
    if request.method == 'POST':
        first_name   = request.POST.get('first_name', '').strip()
        last_name    = request.POST.get('last_name', '').strip()
        username     = request.POST.get('username', '').strip()
        email        = request.POST.get('email', '').strip()
        dept_id      = request.POST.get('department')
        employee_id  = request.POST.get('employee_id', '').strip()
        phone        = request.POST.get('phone', '').strip()
        designation  = request.POST.get('designation', '').strip()
        is_hod       = request.POST.get('is_hod') == 'on'
        dob_str      = request.POST.get('date_of_birth', '').strip()
        selected_subjects = request.POST.getlist('subjects')

        if not first_name or not username or not email or not dept_id:
            messages.error(request, 'First name, username, email and department are required.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered.')
        elif employee_id and User.objects.filter(employee_id=employee_id).exists():
            messages.error(request, f'Employee ID "{employee_id}" is already in use.')
        else:
            from datetime import date
            dob = None
            if dob_str:
                try:
                    dob = date.fromisoformat(dob_str)
                except ValueError:
                    messages.error(request, 'Invalid date of birth.')
                    return render(request, 'admin_panel/teacher_form.html', _teacher_ctx(dept_id))

            dept = get_object_or_404(Department, pk=dept_id)

            # If dept already has a HOD, silently ignore the HOD request
            if is_hod and dept.hod is not None:
                is_hod = False

            if is_hod:
                designation = ''

            pwd = dob_password(dob)
            teacher = User.objects.create(
                username=username, email=email,
                first_name=first_name, last_name=last_name,
                role='teacher', department=dept,
                employee_id=employee_id, phone=phone,
                designation=designation,
                is_hod=is_hod,
                date_of_birth=dob,
                password=make_password(pwd),
                is_first_login=True,
            )
            if is_hod:
                dept.hod = teacher
                dept.save()

            if selected_subjects:
                teacher.assigned_subjects.set(selected_subjects)

            log(request.user, 'create', 'User', f'Onboarded teacher: {teacher.username} in {dept.name}')
            if dob:
                messages.success(request,
                    f'Teacher "{teacher.get_full_name()}" created. '
                    f'Password set to date of birth: <strong>{pwd}</strong>')
            else:
                messages.success(request,
                    f'Teacher "{teacher.get_full_name()}" created. '
                    f'No DOB set — temporary password: <strong>{pwd}</strong>')
            return redirect_back(request, f'/admin-panel/teachers/?dept={dept.pk}')

    selected_dept = None
    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
        except Department.DoesNotExist:
            pass

    dept_semesters = _get_department_semesters(selected_dept)
    return render(request, 'admin_panel/teacher_form.html', {
        'action': 'Add',
        'departments': Department.objects.all(),
        'selected_dept': selected_dept,
        'dept_semesters': dept_semesters,
        'assigned_subject_ids': [],
    })


def _teacher_ctx(dept_id):
    selected_dept = None
    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
        except Department.DoesNotExist:
            pass
    dept_semesters = _get_department_semesters(selected_dept)
    return {
        'action': 'Add',
        'departments': Department.objects.all(),
        'selected_dept': selected_dept,
        'dept_semesters': dept_semesters,
        'assigned_subject_ids': [],
    }


@admin_required
def teacher_edit(request, pk):
    teacher = get_object_or_404(User, pk=pk, role='teacher')
    if request.method == 'POST':
        new_email       = request.POST.get('email', '').strip()
        new_employee_id = request.POST.get('employee_id', '').strip()
        selected_subjects = request.POST.getlist('subjects')

        if new_email and User.objects.filter(email=new_email).exclude(pk=pk).exists():
            messages.error(request, 'Email is already registered to another user.')
            return redirect('teacher_edit', pk=pk)
        if new_employee_id and User.objects.filter(employee_id=new_employee_id).exclude(pk=pk).exists():
            messages.error(request, f'Employee ID "{new_employee_id}" is already in use by another teacher.')
            return redirect('teacher_edit', pk=pk)

        teacher.first_name  = request.POST.get('first_name', '').strip()
        teacher.last_name   = request.POST.get('last_name', '').strip()
        teacher.email       = new_email
        teacher.employee_id = new_employee_id
        teacher.phone       = request.POST.get('phone', '').strip()
        teacher.designation = request.POST.get('designation', '').strip()
        dob_str = request.POST.get('date_of_birth', '').strip()
        if dob_str:
            from datetime import date
            try:
                teacher.date_of_birth = date.fromisoformat(dob_str)
            except ValueError:
                pass

        is_hod   = request.POST.get('is_hod') == 'on'
        dept_id  = request.POST.get('department')
        old_dept = teacher.department

        if dept_id:
            teacher.department = get_object_or_404(Department, pk=dept_id)

        # HOD logic
        was_hod = teacher.is_hod
        new_dept = teacher.department

        if is_hod and not was_hod:
            if new_dept.hod is not None and new_dept.hod != teacher:
                messages.error(request, f'{new_dept.name} already has a HOD.')
                return redirect('teacher_edit', pk=pk)
            new_dept.hod = teacher
            new_dept.save()
            teacher.is_hod = True
        elif not is_hod and was_hod:
            teacher.is_hod = False
            if old_dept and old_dept.hod == teacher:
                old_dept.hod = None
                old_dept.save()

        if teacher.is_hod:
            teacher.designation = ''

        teacher.save()
        teacher.assigned_subjects.set(selected_subjects)

        log(request.user, 'update', 'User', f'Updated teacher: {teacher.username}')
        messages.success(request, f'Teacher "{teacher.get_full_name()}" updated.')
        return redirect_back(request, f'/admin-panel/teachers/?dept={teacher.department_id}')

    dept_semesters = _get_department_semesters(teacher.department)
    assigned_subject_ids = list(teacher.assigned_subjects.values_list('id', flat=True))

    return render(request, 'admin_panel/teacher_form.html', {
        'action': 'Edit',
        'teacher': teacher,
        'departments': Department.objects.all(),
        'selected_dept': teacher.department,
        'dept_semesters': dept_semesters,
        'assigned_subject_ids': assigned_subject_ids,
    })


@admin_required
def api_department_subjects(request):
    from django.http import JsonResponse
    dept_id = request.GET.get('dept')
    data = []
    has_hod = False
    hod_id = None
    hod_name = ''
    if dept_id:
        try:
            dept = Department.objects.select_related('hod').get(pk=dept_id)
            if dept.hod:
                has_hod = True
                hod_id = dept.hod_id
                hod_name = dept.hod.get_full_name() or dept.hod.username
        except Department.DoesNotExist:
            pass
        semesters = Semester.objects.filter(department_id=dept_id).prefetch_related('subjects').order_by('number')
        for sem in semesters:
            subs = [{'id': sub.pk, 'code': sub.subject_code, 'name': sub.name} for sub in sem.subjects.all()]
            data.append({'number': sem.number, 'sem_number': sem.number, 'subjects': subs})
    return JsonResponse({
        'semesters': data,
        'has_hod': has_hod,
        'hod_id': hod_id,
        'hod_name': hod_name,
    })



@admin_required
def teacher_detail(request, pk):
    teacher = get_object_or_404(User, pk=pk, role='teacher')
    # Group assigned subjects by semester
    semesters = Semester.objects.filter(
        department=teacher.department
    ).prefetch_related('subjects').order_by('number')
    assigned_ids = set(teacher.assigned_subjects.values_list('id', flat=True))
    sem_subjects = []
    for sem in semesters:
        subs = [s for s in sem.subjects.all() if s.pk in assigned_ids]
        if subs:
            sem_subjects.append({'semester': sem, 'subjects': subs})
    return render(request, 'admin_panel/teacher_detail.html', {
        'teacher': teacher,
        'sem_subjects': sem_subjects,
    })


@admin_required
def teacher_delete(request, pk):
    teacher = get_object_or_404(User, pk=pk, role='teacher')
    if request.method == 'POST':
        name = teacher.get_full_name() or teacher.username
        dept = teacher.department
        dept_pk = dept.pk if dept else None
        if teacher.is_hod and dept and dept.hod == teacher:
            dept.hod = None
            dept.save()
        teacher.delete()
        log(request.user, 'delete', 'User', f'Deleted teacher: {name}')
        messages.success(request, f'Teacher "{name}" removed.')
        return redirect_back(request, f'/admin-panel/teachers/?dept={dept_pk}' if dept_pk else 'teacher_list')
    return render(request, 'admin_panel/confirm_delete.html', {
        'object_name': teacher.get_full_name() or teacher.username,
        'cancel_url': 'teacher_list'
    })


@admin_required
def teacher_reset_password(request, pk):
    teacher = get_object_or_404(User, pk=pk, role='teacher')
    if request.method == 'POST':
        dept_pk = teacher.department_id
        new_pass = dob_password(teacher.date_of_birth)
        teacher.password = make_password(new_pass)
        teacher.is_first_login = True
        teacher.save()
        log(request.user, 'reset', 'User', f'Password reset for teacher: {teacher.username}')
        if teacher.date_of_birth:
            messages.success(request,
                f'Password reset to date of birth: <strong>{new_pass}</strong>')
        else:
            messages.success(request,
                f'Password reset for "{teacher.get_full_name()}". '
                f'New temporary password: <strong>{new_pass}</strong>')
        return redirect_back(request, f'/admin-panel/teachers/?dept={dept_pk}' if dept_pk else 'teacher_list')
    return render(request, 'admin_panel/confirm_reset.html', {
        'user_obj': teacher, 'cancel_url': 'teacher_list'
    })


# ─── batches ──────────────────────────────────────────────────────────────────

@admin_required
def batch_list(request):
    batches = Batch.objects.select_related('department').order_by('-start_year', '-start_month', '-id')
    departments = Department.objects.all()
    return render(request, 'admin_panel/batch_list.html', {'batches': batches, 'departments': departments})


@admin_required
def batch_add(request):
    if request.method == 'POST':
        name        = request.POST.get('name', '').strip()
        start_month = request.POST.get('start_month', '').strip()
        start_year  = request.POST.get('start_year', '').strip()
        end_month   = request.POST.get('end_month', '').strip()
        end_year    = request.POST.get('end_year', '').strip()
        dept_id     = request.POST.get('department') or request.POST.get('dept')

        if not start_month or not start_year or not end_month or not end_year:
            messages.error(request, 'All date fields are required.')
        else:
            try:
                sm, sy = int(start_month), int(start_year)
                em, ey = int(end_month),   int(end_year)
                if sm < 1 or sm > 12 or em < 1 or em > 12:
                    raise ValueError
                if (sy, sm) >= (ey, em):
                    messages.error(request, 'End date must be after start date.')
                else:
                    dept = None
                    current_sem = None
                    if dept_id:
                        dept = Department.objects.filter(pk=dept_id).first()
                        if dept:
                            current_sem = Semester.objects.filter(department=dept, number=1).first() or Semester.objects.filter(department=dept).order_by('number').first()
                    batch = Batch.objects.create(
                        department=dept,
                        current_semester=current_sem,
                        name=name, start_month=sm, start_year=sy,
                        end_month=em, end_year=ey
                    )
                    dept_label = f" for {dept.name}" if dept else ""
                    log(request.user, 'create', 'Batch', f'Created batch: {batch}{dept_label}')
                    messages.success(request, f'Batch "{batch}" created{dept_label}.')
                    fallback = f'/admin-panel/students/?dept={dept.pk}' if dept else 'batch_list'
                    return redirect_back(request, fallback)
            except ValueError:
                messages.error(request, 'Invalid month/year values.')
    selected_dept = None
    dept_id = request.GET.get('dept')
    if dept_id:
        selected_dept = Department.objects.filter(pk=dept_id).first()
    return render(request, 'admin_panel/batch_form.html', {
        'action': 'Add',
        'departments': Department.objects.all(),
        'selected_dept': selected_dept,
    })


@admin_required
def batch_edit(request, pk):
    batch = get_object_or_404(Batch, pk=pk)
    if request.method == 'POST':
        name        = request.POST.get('name', '').strip()
        start_month = request.POST.get('start_month', '').strip()
        start_year  = request.POST.get('start_year', '').strip()
        end_month   = request.POST.get('end_month', '').strip()
        end_year    = request.POST.get('end_year', '').strip()
        dept_id     = request.POST.get('department') or request.POST.get('dept')

        if not start_month or not start_year or not end_month or not end_year:
            messages.error(request, 'All date fields are required.')
        else:
            try:
                sm, sy = int(start_month), int(start_year)
                em, ey = int(end_month),   int(end_year)
                if sm < 1 or sm > 12 or em < 1 or em > 12:
                    raise ValueError
                if (sy, sm) >= (ey, em):
                    messages.error(request, 'End date must be after start date.')
                else:
                    if dept_id:
                        dept = Department.objects.filter(pk=dept_id).first()
                        if dept:
                            batch.department = dept
                    batch.name = name
                    batch.start_month = sm
                    batch.start_year  = sy
                    batch.end_month   = em
                    batch.end_year    = ey
                    batch.save()
                    log(request.user, 'update', 'Batch', f'Updated batch: {batch}')
                    messages.success(request, f'Batch "{batch}" updated successfully.')
                    fallback = f'/admin-panel/students/?dept={batch.department_id}' if batch.department_id else 'batch_list'
                    return redirect_back(request, fallback)
            except ValueError:
                messages.error(request, 'Invalid month/year values.')
    fallback = f'/admin-panel/students/?dept={batch.department_id}' if batch.department_id else 'batch_list'
    return redirect_back(request, fallback)


@admin_required
def batch_delete(request, pk):
    batch = get_object_or_404(Batch, pk=pk)
    dept_pk = batch.department_id
    if request.method == 'POST':
        name = str(batch)
        batch.delete()
        log(request.user, 'delete', 'Batch', f'Deleted batch: {name}')
        messages.success(request, f'Batch "{name}" deleted.')
        fallback = f'/admin-panel/students/?dept={dept_pk}' if dept_pk else 'batch_list'
        return redirect_back(request, fallback)
    return render(request, 'admin_panel/confirm_delete.html', {
        'object_name': str(batch), 'cancel_url': 'batch_list'
    })


# ─── students ─────────────────────────────────────────────────────────────────

@admin_required
def student_list(request):
    """
    Three-level drill-down: Department → Batch → Semester → Students table.
    Query params: dept, batch, sem
    """
    dept_id  = request.GET.get('dept')
    batch_id = request.GET.get('batch')
    sem_id   = request.GET.get('sem')

    departments    = Department.objects.annotate(
        student_count=Count('users', filter=Q(users__role='student'))
    ).order_by('name')
    batches        = []
    semesters      = []
    selected_dept  = None
    selected_batch = None
    selected_sem   = None
    next_sem       = None
    students       = None

    # Backward compatibility: if batch_id provided without dept_id, allow finding batch
    if batch_id and not dept_id:
        try:
            selected_batch = Batch.objects.get(pk=batch_id)
            if selected_batch.department:
                selected_dept = selected_batch.department
        except Batch.DoesNotExist:
            pass

    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
            # Batches available with student count in this specific department (newest first)
            batches = Batch.objects.filter(department=selected_dept).annotate(
                student_count=Count('students', filter=Q(students__department=selected_dept, students__role='student'))
            ).order_by('-start_year', '-start_month', '-id')
        except Department.DoesNotExist:
            pass

    if selected_dept and batch_id:
        try:
            selected_batch = Batch.objects.get(pk=batch_id, department=selected_dept)
            active_sem = selected_batch.current_semester
            if not active_sem:
                active_sem = (
                    Semester.objects.filter(students__batch=selected_batch, students__role='student').order_by('-number').first()
                    or Semester.objects.filter(department=selected_dept, number=1).first()
                    or Semester.objects.filter(department=selected_dept).order_by('number').first()
                )
                if active_sem:
                    selected_batch.current_semester = active_sem
                    selected_batch.save(update_fields=['current_semester'])

            if active_sem:
                semesters = Semester.objects.filter(pk=active_sem.pk).annotate(
                    student_count=Count('students', filter=Q(students__batch=selected_batch, students__role='student'))
                )
            else:
                semesters = []
        except Batch.DoesNotExist:
            pass

    if selected_dept and selected_batch and sem_id:
        try:
            selected_sem = Semester.objects.get(pk=sem_id, department=selected_dept)
            students = User.objects.filter(
                role='student',
                batch=selected_batch,
                department=selected_dept,
                semester=selected_sem,
            ).select_related('department', 'semester')
            next_sem = Semester.objects.filter(
                department=selected_dept,
                number=selected_sem.number + 1
            ).first()
        except Semester.DoesNotExist:
            pass

    return render(request, 'admin_panel/student_list.html', {
        'departments': departments,
        'selected_dept': selected_dept,
        'batches': batches,
        'selected_batch': selected_batch,
        'semesters': semesters,
        'selected_sem': selected_sem,
        'next_sem': next_sem,
        'students': students,
    })


@admin_required
def student_add(request):
    batch_id = request.GET.get('batch') or request.POST.get('batch')
    dept_id  = request.GET.get('dept')  or request.POST.get('department')
    sem_id   = request.GET.get('sem')   or request.POST.get('semester')

    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name  = request.POST.get('last_name', '').strip()
        username   = request.POST.get('username', '').strip()
        email      = request.POST.get('email', '').strip()
        dob_str    = request.POST.get('date_of_birth', '').strip()
        batch_id   = request.POST.get('batch')
        dept_id    = request.POST.get('department')
        sem_id     = request.POST.get('semester')

        if not first_name or not username or not dept_id or not sem_id or not batch_id:
            messages.error(request, 'First name, register number, department, semester and batch are required.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Register number / username already registered.')
        elif email and User.objects.filter(email=email).exists():
            messages.error(request, 'Email is already registered to another user.')
        else:
            from datetime import date
            dob = None
            if dob_str:
                try:
                    dob = date.fromisoformat(dob_str)
                except ValueError:
                    messages.error(request, 'Invalid date of birth.')
                    return _student_add_render(request, batch_id, dept_id, sem_id)

            dept    = get_object_or_404(Department, pk=dept_id)
            sem     = get_object_or_404(Semester, pk=sem_id)
            batch   = get_object_or_404(Batch, pk=batch_id)
            pwd     = dob_password(dob)
            student = User.objects.create(
                username=username, email=email,
                first_name=first_name, last_name=last_name,
                role='student', department=dept,
                semester=sem,
                batch=batch,
                date_of_birth=dob,
                password=make_password(pwd),
                is_first_login=True,
            )
            log(request.user, 'create', 'User', f'Registered student: {student.username} in {dept.name}')
            if dob:
                messages.success(request,
                    f'Student "{student.get_full_name()}" added. '
                    f'Password set to date of birth: <strong>{pwd}</strong>')
            else:
                messages.success(request,
                    f'Student "{student.get_full_name()}" added. '
                    f'No DOB — temporary password: <strong>{pwd}</strong>')
            next_url = request.POST.get('next') or request.GET.get('next')
            if next_url and '/students/add/' not in next_url:
                return redirect(next_url)
            return redirect(f'/admin-panel/students/?dept={dept.pk}&batch={batch.pk}&sem={sem.pk}')



    return _student_add_render(request, batch_id, dept_id, sem_id)


def _student_add_render(request, batch_id, dept_id, sem_id):
    selected_dept  = None
    selected_batch = None
    selected_sem   = None
    if dept_id:
        try:
            selected_dept = Department.objects.get(pk=dept_id)
        except Department.DoesNotExist:
            pass
    if batch_id:
        try:
            selected_batch = Batch.objects.get(pk=batch_id)
            if not selected_sem and selected_batch.current_semester:
                selected_sem = selected_batch.current_semester
        except Batch.DoesNotExist:
            pass
    if sem_id:
        try:
            selected_sem = Semester.objects.get(pk=sem_id)
        except Semester.DoesNotExist:
            pass

    departments = Department.objects.all().order_by('name')
    if selected_dept:
        batches = Batch.objects.filter(department=selected_dept).order_by('-start_year', '-start_month', '-id')
        semesters = Semester.objects.filter(department=selected_dept).order_by('number')
    else:
        batches = Batch.objects.order_by('-start_year', '-start_month', '-id')
        semesters = []

    return render(request, 'admin_panel/student_form.html', {
        'action': 'Add',
        'selected_dept': selected_dept,
        'selected_batch': selected_batch,
        'selected_sem': selected_sem,
        'departments': departments,
        'batches': batches,
        'semesters': semesters,
    })


@admin_required
def student_edit(request, pk):
    student = get_object_or_404(User, pk=pk, role='student')
    if request.method == 'POST':
        new_email = request.POST.get('email', '').strip()

        if new_email and User.objects.filter(email=new_email).exclude(pk=pk).exists():
            messages.error(request, 'Email is already registered to another user.')
            return redirect('student_edit', pk=pk)

        student.first_name = request.POST.get('first_name', '').strip()
        student.last_name  = request.POST.get('last_name', '').strip()
        student.email      = new_email
        dob_str = request.POST.get('date_of_birth', '').strip()
        if dob_str:
            from datetime import date
            try:
                student.date_of_birth = date.fromisoformat(dob_str)
            except ValueError:
                pass
        dept_id  = request.POST.get('department')
        batch_id = request.POST.get('batch')
        sem_id   = request.POST.get('semester')
        if dept_id:
            student.department = get_object_or_404(Department, pk=dept_id)
        if batch_id:
            student.batch = get_object_or_404(Batch, pk=batch_id)
        if sem_id:
            student.semester = get_object_or_404(Semester, pk=sem_id)
        student.save()
        log(request.user, 'update', 'User', f'Updated student: {student.username}')
        messages.success(request, f'Student "{student.get_full_name()}" updated.')
        b = student.batch_id or ''
        d = student.department_id or ''
        s = student.semester_id or ''
        next_url = request.POST.get('next') or request.GET.get('next')
        if next_url and f'/students/{pk}/edit/' not in next_url:
            return redirect(next_url)
        return redirect(f'/admin-panel/students/?dept={d}&batch={b}&sem={s}')
    return render(request, 'admin_panel/student_form.html', {
        'action': 'Edit',
        'student': student,
        'selected_dept': student.department,
        'selected_batch': student.batch,
        'selected_sem': student.semester,
        'departments': Department.objects.all().order_by('name'),
        'batches': Batch.objects.filter(department=student.department).order_by('-start_year', '-start_month', '-id') if student.department else Batch.objects.order_by('-start_year', '-start_month', '-id'),
        'semesters': Semester.objects.filter(department=student.department).order_by('number') if student.department else [],
    })


@admin_required
def student_delete(request, pk):
    student = get_object_or_404(User, pk=pk, role='student')
    if request.method == 'POST':
        name = student.get_full_name() or student.username
        b = student.batch_id or ''
        d = student.department_id or ''
        s = student.semester_id or ''
        student.delete()
        log(request.user, 'delete', 'User', f'Deleted student: {name}')
        messages.success(request, f'Student "{name}" removed.')
        return redirect_back(request, f'/admin-panel/students/?dept={d}&batch={b}&sem={s}')
    return render(request, 'admin_panel/confirm_delete.html', {
        'object_name': student.get_full_name() or student.username,
        'cancel_url': 'student_list'
    })


@admin_required
def student_reset_password(request, pk):
    student = get_object_or_404(User, pk=pk, role='student')
    if request.method == 'POST':
        b = student.batch_id or ''
        d = student.department_id or ''
        s = student.semester_id or ''
        new_pass = dob_password(student.date_of_birth)
        student.password = make_password(new_pass)
        student.is_first_login = True
        student.save()
        log(request.user, 'reset', 'User', f'Password reset for student: {student.username}')
        if student.date_of_birth:
            messages.success(request, f'Password reset to date of birth: <strong>{new_pass}</strong>')
        else:
            messages.success(request,
                f'Password reset for "{student.get_full_name()}". '
                f'New temporary password: <strong>{new_pass}</strong>')
        fallback = f'/admin-panel/students/?dept={d}&batch={b}&sem={s}' if d else 'student_list'
        return redirect_back(request, fallback)
    return render(request, 'admin_panel/confirm_reset.html', {
        'user_obj': student, 'cancel_url': 'student_list'
    })


@admin_required
def student_promote(request):
    if request.method == 'POST':
        batch_id = request.POST.get('batch')
        dept_id  = request.POST.get('department')
        sem_id   = request.POST.get('semester')

        if not batch_id or not dept_id or not sem_id:
            messages.error(request, 'Batch, Department, and Semester are required for promotion.')
            return redirect_back(request, 'student_list')

        batch = get_object_or_404(Batch, pk=batch_id)
        dept  = get_object_or_404(Department, pk=dept_id)
        current_sem = get_object_or_404(Semester, pk=sem_id, department=dept)

        # Find the next sequential semester in this department
        next_sem = Semester.objects.filter(
            department=dept,
            number=current_sem.number + 1
        ).first()

        students = User.objects.filter(
            role='student',
            batch=batch,
            department=dept,
            semester=current_sem
        )
        student_count = students.count()

        if student_count == 0:
            messages.warning(request, f'No students found in Semester {current_sem.number} for Batch "{batch}".')
            return redirect_back(request, f'/admin-panel/students/?dept={dept.pk}&batch={batch.pk}&sem={current_sem.pk}')

        if not next_sem:
            messages.warning(
                request,
                f'Semester {current_sem.number} is the highest configured semester for {dept.name}. '
                f'To advance these students, please add Semester {current_sem.number + 1} under Academic Roadmap ➔ Semesters.'
            )
            return redirect_back(request, f'/admin-panel/students/?dept={dept.pk}&batch={batch.pk}&sem={current_sem.pk}')

        # Perform bulk promotion to next semester
        students.update(semester=next_sem)
        batch.current_semester = next_sem
        batch.save(update_fields=['current_semester'])
        log(
            request.user, 'update', 'User',
            f'Promoted {student_count} student(s) of Batch "{batch}" ({dept.code}) from Semester {current_sem.number} to Semester {next_sem.number}'
        )
        messages.success(
            request,
            f'Successfully promoted {student_count} student(s) from Semester {current_sem.number} to Semester {next_sem.number}!'
        )
        return redirect(f'/admin-panel/students/?dept={dept.pk}&batch={batch.pk}&sem={next_sem.pk}')

    return redirect_back(request, 'student_list')


# ─── audit log ────────────────────────────────────────────────────────────────

@admin_required
def audit_log(request):
    logs = AuditLog.objects.select_related('actor').all()
    action_filter = request.GET.get('action')
    if action_filter:
        logs = logs.filter(action=action_filter)
    return render(request, 'admin_panel/audit_log.html', {
        'logs': logs[:200],
        'action_filter': action_filter,
    })





