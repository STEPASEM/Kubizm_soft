from django.urls import path

from . import views

app_name = 'main_page'

urlpatterns = [
    path('', views.Index, name='index'),
    path('o-we/', views.o_we, name='o_we'),
]
