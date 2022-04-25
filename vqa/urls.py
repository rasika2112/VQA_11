
from django.urls import path, include
#now import the views.py file into this code
from . import views
urlpatterns=[
  path('',views.index, name="index"),
  path('upload', views.upload, name="upload"),
  path('captcha', views.captcha, name="captcha"),
  path('captcha_api', views.captcha_api, name="captcha_api"),
  path('api-auth/', include('rest_framework.urls')),
  path('logout', views.logout, name="logout"),
  path('about', views.about, name="about")
]