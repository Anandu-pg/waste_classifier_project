from django.apps import AppConfig

class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    def ready(self):
        from . import signals  # Correct way to import signals
