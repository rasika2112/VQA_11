from ast import Pass
from asyncio.windows_events import NULL
import email
import json
from pickle import GET
from unittest import result
from urllib import response
from django.shortcuts import render, redirect
from django.http import HttpResponse
import pyrebase
import random, ast
from .vqa_code import vqa_func 
import requests
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from rest_framework.response import Response
from rest_framework.decorators import api_view

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

admin = "admin"
adminp = "admin"

# Admin login page to proceed to video uploading page
def index(request):
    if request.method =="POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        print(username, password)
        if username==admin and password == adminp:
            request.session['Message'] = ""
            request.session['Username'] = username
            print(request.session['Username'])
            return redirect('upload')
        else:
            request.session['Message'] = "Incorrect Username or Password"
            request.session['Username'] == NULL
            return redirect('/')
    else:
        return render(request, 'index.html')

# CAPTCHA verification after user's post request with its answer to our application
@api_view(['POST'])
def captcha(request):
    if request.method =="POST":
        answer = request.data['answer']
        video_name = request.data["video_name"]
        question_number = int(request.data["question_number"])
        Video = database.child('Videos').get().val()
        print(Video)
        correct_answer = database.child('Videos').child(video_name).child(question_number).child('Answer').get().val()
        print(correct_answer == answer, correct_answer, answer)
        if correct_answer == answer:    
            print(answer)
            respons = {'result': 'success'}
        else:
            request.session['Message'] = "Incorrect CAPTCHA"
            respons = {'result': 'failure'}
        request.session['Username'] = NULL
        print('captcha evaluated')
        return HttpResponse(json.dumps(respons))
    else:
        return HttpResponse()

# Uploading of video for MCQ generation by the admin
def upload(request):
    if "Username" in request.session:
        if request.method == "POST":
            title = request.POST.get("title")
            file = request.FILES['file']
            storage.child('/' + title).put(file)
            url = "https://firebasestorage.googleapis.com/v0/b/vqa11-fccb4.appspot.com/o/" + title
            data = requests.get(url=url).json()
            print(data)
            video_path = url + '?alt=media&token=' + data['downloadTokens']
            result = vqa_func(title, video_path)
            print(result)
            if result==None:
                request.session["Upload_Message"] = "File upload with questions in Firebase Storage unsuccessful"
            else:
                result['Link'] = video_path
                database.child('Videos/' + title).set(result)
                request.session["Upload_Message"] = "File upload with questions in Firebase Storage successful"
            return redirect('upload')
        else:
            request.session['Message'] = ""
            return render(request, "upload.html")
    else:
        request.session['Message'] = "Please Login"
        return redirect('/')

# Logout from the current session
def logout(request):
    del request.session["Username"]
    if "Message" in request.session:
        del request.session["Message"]
    if "Upload_Message" in request.session:
        del request.session["Upload_Message"]
    return redirect('/')

# About page of our application
def about(request):
    return render(request, 'about.html')

# API called by the third party applications for gaining the video CAPTCHA details shown to the user
@api_view(['POST'])
def captcha_api(request):
    data = request.data
    print(data)
    if 'username' in data.keys():
        request.session['Username'] = data['username']
        Video_data = database.child('Videos').get().val()
        users = list(dict(database.child('Users').get().val()).keys())
        if data['username'] not in users:
            database.child('Users/' + data['username']).set('Videos')
        watched_videos = database.child('Users').child(data['username']).child('Videos').get().val()
        watched_videos_list = watched_videos
        if watched_videos!=None:
            if len(Video_data)!=len(watched_videos):
                for video in watched_videos:
                    del Video_data[video]
            else:
                watched_videos_list = []
        else:
            watched_videos_list = []
        Video= random.choice(list(dict(Video_data).items()))
        video_name = Video[0]
        video_details = Video[1]
        watched_videos_list.append(video_name)
        database.child('Users/' + data['username'] + '/Videos').set(watched_videos_list)
        video_path = video_details['Link']
        del video_details['Link']
        question_number = random.choice(list(video_details.items()))
        question_details = question_number[1]
        print(question_details)
        question = question_details['Question']
        choices = question_details['Choices']
        res = {'Question': question, 'Choices': choices}
        result = { 'video_path': video_path, 
                'res': res, 
                'video_name': video_name, 
                'number': question_number[0]}
        print('captcha shown')
        return HttpResponse(json.dumps(result))
    else:
        return HttpResponse()