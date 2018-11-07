from django.urls import path
from . import views

app_name = 'community'
urlpatterns = [
    # /community/
    path('', views.index, name='index'),
    path('discover', views.discover, name='discover'),
    path('discover/<int:community_tag>', views.detail, name='detail')
]
