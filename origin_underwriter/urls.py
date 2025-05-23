from django.contrib import admin
from django.urls import path, include
from rest_framework.authtoken.views import obtain_auth_token


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/documents/', include('documents.urls')),  # Add this line
    path('api/v1/login/', obtain_auth_token, name='api_token_auth'),
]