from ast import Pass
from asyncio.windows_events import NULL
import email
from pickle import GET
from unittest import result
from django.shortcuts import render, redirect
from django.http import HttpResponse
import pyrebase
import random, ast
from .vqa_code import vqa_func 
import requests
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage

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
storage = firebase.storage()
# bucket = storage.bucket()


def index(request):
    if request.method =="POST":
        Users = list(database.child('Users').get().val())
        print(Users)
        username = request.POST.get("username")
        password = request.POST.get("password")
        correct_password = database.child('Users').child(username).child('Password').get().val()
        print(username in Users)
        if username in Users and password == correct_password:
            request.session['Username'] = username
            print(request.session['Username'])
            return redirect('captcha')
        else:
            request.session['Message'] = "Incorrect Username or Password"
            request.session['Username'] == NULL
            return redirect('/')
    else:
        return render(request, 'index.html')


def captcha(request):
    if request.method =="POST":
        answer = request.POST.get("answer")
        video_name = request.POST.get("video_name")
        question_number = int(request.POST.get("number"))
        Video = database.child('Videos').get().val()
        print(Video)
        # print(video_name)
        video_path = database.child('Videos').child(video_name).child('Link').get().val()
        question = database.child('Videos').child(video_name).child(question_number).child('Question').get().val()
        choices = database.child('Videos').child(video_name).child(question_number).child('Choices').get().val()
        correct_answer = database.child('Videos').child(video_name).child(question_number).child('Answer').get().val()
        res = {'Question': question, 'Choices': choices}
        print(correct_answer == answer, correct_answer, answer)
        if correct_answer == answer:    
            print(answer)
        else:
            request.session['Message'] = "Incorrect CAPTCHA"
            return redirect('/')
        request.session['Username'] = NULL
        return render(request, 'success.html')
    else:
        Video_data = database.child('Videos').get().val()
        # print(Video_data)
        if request.session['Username'] == NULL:
            request.session['Message'] = "Please enter credentials"
            return redirect('/')
        watched_videos = database.child('Users').child(request.session['Username']).child('Videos').get().val()
        watched_videos_list = watched_videos
        print(len(Video_data), len(watched_videos))
        if len(Video_data)!=len(watched_videos):
            for video in watched_videos:
                del Video_data[video]
        else:
            watched_videos_list = []
        Video= random.choice(list(dict(Video_data).items()))
        # print(Video)
        video_name = Video[0]
        video_details = Video[1]
        watched_videos_list.append(video_name)
        database.child('Users/' + request.session['Username'] + '/Videos').set(watched_videos_list)
        # print(video_name)
        print(video_details)
        video_path = video_details['Link']
        del video_details['Link']
        question_number = random.choice(list(video_details.items()))
        question_details = question_number[1]
        print(question_details)
        question = question_details['Question']
        choices = question_details['Choices']
        print(choices)
        res = {'Question': question, 'Choices': choices}
        # print(res)
        return render(request, 'captcha.html', {'video_path': video_path, 'res': res, 'video_name': video_name, 'number': question_number[0]})


def upload(request):
    if request.method == "POST":
        title = request.POST.get("title")
        file = request.FILES['file']
        # path = default_storage.save(title, file)
        storage.child('/' + title).put(file)
        # print(default_storage.delete(path))
        url = "https://firebasestorage.googleapis.com/v0/b/vqa11-fccb4.appspot.com/o/" + title
        data = requests.get(url=url).json()
        print(data)
        video_path = url + '?alt=media&token=' + data['downloadTokens']
        result = vqa_func(video_path)
        # result = {'Question': 'What is man in blue shirt riding on crowd?', 'Choices': ['Skateboard', 'unicycle', 'Tricycle', 'Riding'], 'Answer': 'unicycle'}
        if result==NULL:
            print("No questions generated")
        else:
            result['Link'] = video_path
            database.child('Videos/' + title).set(result)
            print( "File upload in Firebase Storage successful")
        return redirect('index')
    else:
        return render(request, "upload.html")