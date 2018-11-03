from django.urls import path
from . import views

app_name = 'music'
urlpatterns = [
    # /music/
    path('', views.IndexView.as_view(), name='index'),

    # /music/<album_id>/
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),

    # /music/<album_id>/favorite
    path('<int:album_id>/favorite/', views.favorite, name='favorite'),
    path('album/add/', views.AlbumCreate.as_view(), name='album-add')
]
