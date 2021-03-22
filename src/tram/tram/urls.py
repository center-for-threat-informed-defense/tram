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
from rest_framework.routers import DefaultRouter

from tram import views


router = DefaultRouter()
router.register(r'attack', views.AttackTechniqueViewSet)
router.register(r'jobs', views.DocumentProcessingJobViewSet)
router.register(r'mappings', views.MappingViewSet)
router.register(r'reports', views.ReportViewSet)
router.register(r'sentences', views.SentenceViewSet)


urlpatterns = [
    path('', views.index),
    path('analyze/<int:pk>/', views.analyze),
    path('api/', include(router.urls)),
    path('login/', auth_views.LoginView.as_view()),
    path('logout/', auth_views.LogoutView.as_view()),
    path('upload/', views.upload),
    path('admin/', admin.site.urls),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
