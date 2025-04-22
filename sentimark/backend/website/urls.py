from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path('logout/', views.logout_view, name='logout'),
    path("login", views.login_view, name="login"),
    path("signup", views.signup, name="signup"),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('profile/', views.profile, name='profile'),
    path("analyzed_results/<int:sentiment_data_id>", views.analyzed_results, name="analyzed_results"),
]