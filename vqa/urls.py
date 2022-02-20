
from django.urls import path
#now import the views.py file into this code
from . import views
urlpatterns=[
  path('',views.index, name="index"),
  path('upload', views.upload, name="upload"),
  path('captcha', views.captcha, name="captcha")
]