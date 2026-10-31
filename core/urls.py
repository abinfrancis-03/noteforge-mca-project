from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from accounts import views as auth_views

urlpatterns = [
    path('django-admin/', admin.site.urls),
    path('login/',  auth_views.login_view,  name='login'),
    path('logout/', auth_views.logout_view, name='logout'),
    path('admin-panel/', include('admin_panel.urls')),
    path('teacher/', include('teacher_panel.urls')),
    path('student/', include('student_panel.urls')),
    path('', auth_views.home_redirect, name='home'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

