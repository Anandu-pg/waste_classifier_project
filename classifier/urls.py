from django.urls import path
from . import views

urlpatterns = [
    path('', views.intro, name='intro'),
    path('home/', views.classify, name='home'),
    path('agenda/', views.agenda, name='agenda'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("chat/", views.chat_bot, name="chat"),
    path('get_bot_response/', views.get_bot_response, name='get_bot_response'),

]
