from ast import Pass
from pickle import GET
from django.shortcuts import render
from django.http import HttpResponse
import pyrebase

config = {
        "apiKey": "AIzaSyDZ8DZQb7ulhRh6CZJQh4YIr_6U9cSi_aA",
        "authDomain": "vqa11-fccb4.firebaseapp.com",
        "databaseURL": "https://vqa11-fccb4-default-rtdb.firebaseio.com",
        "projectId": "vqa11-fccb4",
        "storageBucket": "vqa11-fccb4.appspot.com",
        "messagingSenderId": "917773394703",
        "appId": "1:917773394703:web:6605dba8619ed9def1547a"
      }

firebase = pyrebase.initialize_app(config)
authe = firebase.auth()
database = firebase.database()


def index(request):
    if request.method =="POST":
        Users_data = database.child('Users').get().val()
        Users = dict(Users_data.items())
        # print(Users)
        username = request.POST.get("username")
        password = request.POST.get("password")
        answer = request.POST.get("answer")
        if username in Users.keys() and password == Users[username]:
            print(answer) 
        return render(request, 'index.html')
    else:
         return render(request, 'index.html')
