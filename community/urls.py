from django.urls import path
from . import views
from . import views2

app_name = 'community'
urlpatterns = [
    # /community/
    path('', views.index, name='index'),
    path('discover', views.discover, name='discover'),
    path('discover/<int:community_tag>', views.detail, name='detail'),

    path('v2', views2.index, name='index2'),
    path('v2/discover', views2.discover, name='discover2'),
    path('v2/discover/<int:community_tag>', views2.detail, name='detail2'),
]
