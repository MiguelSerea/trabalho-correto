from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('centralized-uniprocess/', views.start_training, name='start_training'),
    path('centralized-multiprocess/', views.centralized_multiprocess_view, name='centralized_multiprocess'),
    path('descentralized-multiprocess/', views.start_combinations, name='descentralized-multiprocess')
]