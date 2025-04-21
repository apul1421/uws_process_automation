from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import CustomerDocumentUploadViewSet

router = DefaultRouter()
router.register(r'upload', CustomerDocumentUploadViewSet, basename='customer-upload')

urlpatterns = router.urls