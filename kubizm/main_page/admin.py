from django.contrib import admin
from .models import Artwork

@admin.register(Artwork)
class ArtworkAdmin(admin.ModelAdmin):
    list_display = ['title', 'artist', 'style', 'year']
    list_filter = ['style']
    search_fields = ['title', 'artist']