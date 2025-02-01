from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('start-training/', views.start_training, name='start_training'),
    path('centralized-multiprocess/', views.centralized_multiprocess_view, name='centralized_multiprocess'),
    path('register-worker/', views.register_worker, name='register_worker'),
    path('processRequest/', views.processRequest, name='processRequest'),
    path('combine/', views.combine, name='combine'),
]