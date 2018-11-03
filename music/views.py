from .models import Album, Song
from django.shortcuts import render, get_object_or_404
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views import generic


class IndexView(generic.ListView):
    template_name = 'index.html'
    context_object_name = 'all_albums'

    def get_queryset(self):
        return Album.objects.all()


class DetailView(generic.DetailView):
    model = Album
    template_name = 'detail.html'


def favorite(request, album_id):
    album = get_object_or_404(Album, pk=album_id)
    try:
        selected_songs = []
        for song_id in request.POST.getlist('song'):
            selected_songs.append(album.song_set.get(pk=song_id))
    except (KeyError, Song.DoesNotExist):
        return render(request, 'detail.html',
                      {'album': album, 'error_message': 'You did not select a valid song.'})
    else:
        for song in selected_songs:
            if 'like' in request.POST:
                song.is_favorite = True
            elif 'dislike' in request.POST:
                song.is_favorite = False
            song.save()
        return render(request, 'detail.html', {'album': album})


class AlbumCreate(CreateView):
    model = Album
    template_name = 'album_form.html'
    fields = {'artist', 'album_title', 'genre', 'album_logo'}
