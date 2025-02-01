from django.apps import AppConfig
from threading import Timer
from .views import send_tasks_to_workers

class InterfaceConfig(AppConfig):
    name = 'interface'

    def ready(self):
        # Inicia o envio de tarefas para os workers
        Timer(5.0, send_tasks_to_workers).start()