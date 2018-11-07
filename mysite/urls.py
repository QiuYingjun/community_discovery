from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('music/', include('music.urls')),
    # path('', include('music.urls')),
    path('',include('community.urls'))
]
