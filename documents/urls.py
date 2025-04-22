from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import CustomerDocumentUploadViewSet

router = DefaultRouter()
router.register(r'', CustomerDocumentUploadViewSet, basename='customer-upload')  # ✅ empty string here

urlpatterns = router.urls