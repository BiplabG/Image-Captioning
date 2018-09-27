from django.urls import path
from . import views

urlpatterns = [
     path('', views.index, name = 'index'),
     path('caption/<img_id>', views.display_caption, name='image_with_caption'),
]
