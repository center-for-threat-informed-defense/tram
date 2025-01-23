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
from django.urls import include, path
from django.views.generic.base import TemplateView
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from tram import views

router = DefaultRouter()
router.register(r"attack", views.AttackObjectViewSet)
router.register(r"jobs", views.DocumentProcessingJobViewSet)
router.register(r"mappings", views.MappingViewSet)
router.register(r"reports", views.ReportViewSet)
router.register(
    r"report-mappings", views.ReportMappingViewSet, basename="report-mapping"
)
router.register(r"sentences", views.SentenceViewSet)


urlpatterns = [
    path("", views.index),
    path("analyze/<int:pk>/", views.analyze),
    path("api/", include(router.urls)),
    path("api/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/upload/", views.upload_api),
    path("api/download/<int:doc_id>", views.download_document),
    path("api/train-model/<name>", views.train_model),
    path("docs/", TemplateView.as_view(template_name="tram_documentation.html")),
    path("login/", auth_views.LoginView.as_view()),
    path("logout/", auth_views.LogoutView.as_view()),
    path("upload/", views.upload_web),
    path("admin/", admin.site.urls),
    path("ml/", views.ml_home),
    path("ml/techniques/<str:attack_id>", views.ml_technique_sentences),
    path("ml/models/<str:model_key>", views.ml_model_detail),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
