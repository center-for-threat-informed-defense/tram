"""tram URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.views.generic.base import TemplateView
from rest_framework.routers import DefaultRouter

from tram import views


router = DefaultRouter()
router.register(r'attack', views.AttackObjectViewSet)
router.register(r'jobs', views.DocumentProcessingJobViewSet)
router.register(r'mappings', views.MappingViewSet)
router.register(r'reports', views.ReportViewSet)
router.register(r'report-export', views.ReportExportViewSet)
router.register(r'sentences', views.SentenceViewSet)


urlpatterns = [
    path('', views.index),
    path('analyze/<int:pk>/', views.analyze),
    path('api/', include(router.urls)),
    path('api/download/<str:report_name>', views.download_report),
    path('docs/', TemplateView.as_view(template_name='tram_documentation.html')),
    path('login/', auth_views.LoginView.as_view()),
    path('logout/', auth_views.LogoutView.as_view()),
    path('upload/', views.upload),
    path('admin/', admin.site.urls),
    path('ml/', views.ml_home),
    path('ml/techniques/<str:attack_id>', views.ml_technique_sentences),
    path('ml/models/<str:model_key>', views.ml_model_detail),
    path('ml/retrain/<str:model_key>', views.ml_model_retrain)
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
