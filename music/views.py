from .models import Album, Song
from django.shortcuts import render, get_object_or_404


def index(request):
    all_albums = Album.objects.all()
    context = {'all_albums': all_albums, }
    return render(request, 'music/index.html', context)


def detail(request, album_id):
    album = get_object_or_404(Album, pk=album_id)
    return render(request, 'music/detail.html', {'album': album})


def favorite(request, album_id):
    album = get_object_or_404(Album, pk=album_id)
    try:
        selected_songs = []
        for song_id in request.POST.getlist('song'):
            selected_songs.append(album.song_set.get(pk=song_id))
    except (KeyError, Song.DoesNotExist):
        return render(request, 'music/detail.html',
                      {'album': album, 'error_message': 'You did not select a valid song.'})
    else:
        for song in selected_songs:
            if 'like' in request.POST:
                song.is_favorite = True
            elif 'dislike' in request.POST:
                song.is_favorite = False
            song.save()
        return render(request, 'music/detail.html', {'album': album})

