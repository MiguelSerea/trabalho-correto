from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('start-training/', views.start_training, name='start_training'),
    path('centralized-multiprocess/', views.centralized_multiprocess_view, name='centralized_multiprocess'),
    path('register-worker/', views.register_worker, name='register_worker'),
    path('decentralized_multiprocess/', views.decentralized_multiprocess_view, name='decentralized_multiprocess'),
    path('combine/', views.combine, name='combine'),
    path('record_results/', views.record_results, name='record_results'),
]


