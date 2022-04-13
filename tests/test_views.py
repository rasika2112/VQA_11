from urllib import response
from django.test import TestCase, Client
from django.urls import reverse
import json

class TestViews(TestCase):

    def setUp(self):
        self.client = Client()
        self.index_url = reverse('index')
        self.upload_url = reverse('upload') 
        self.captcha_url = reverse('captcha')
        self.captcha_api_url = reverse('captcha_api')

    def test_index_GET(self):
        response = self.client.get(self.index_url)

        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'index.html')

    def test_upload_GET(self):
        response = self.client.get(self.upload_url)

        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'upload.html')


    # Correct call to captcha api for details
    # Requesting video captcha details by passing the credentials of the user
    def test_captcha_api_correct_POST(self):

        # Passing the username and requesting captcha details
        response = self.client.post(self.captcha_api_url, {
            'username': 'rasika2112'
        })

        # Asserting the status code and content
        self.assertEquals(response.status_code, 200)
        # r = json.loads(response.content)

        # Asserting that response is not empty and contains the details
        self.assertNotEquals(response.content, b'')


    # Incorrect call to captcha api for details
    # Requesting video captcha details by not passing the credentials of the user
    def test_captcha_api_wrong_POST(self):

        # Not passing the username and requesting captcha details
        response = self.client.post(self.captcha_api_url, {
        })

        # Asserting the status code and content
        self.assertEquals(response.status_code, 200)

        # Asserting that response is empty and doesn't contains the details
        self.assertEquals(response.content, b'')


    # Checking the result when user has answered correctly
    def test_captcha_CorrectAnswer_POST(self):

        # Passing the details of question and user's correct answer
        response = self.client.post(self.captcha_url, {
            'answer': 'skateboard',
            'video_name': 'Sample 1',
            'question_number': 2
        })

        # Asserting the status code and content
        self.assertEquals(response.status_code, 200)
        r = json.loads(response.content)

        # Asserting the correct answer and system's answer with result as success
        self.assertEquals(r['result'], 'success')


    # Checking the result when user has answered incorrectly
    def test_captcha_WrongAnswer_POST(self):

        # Passing the details of question and user's incorrect answer
        response = self.client.post(self.captcha_url, {
            'answer': 'bike',
            'video_name': 'Sample 1',
            'question_number': 2
        })

        # Asserting the status code and content
        self.assertEquals(response.status_code, 200)
        r = json.loads(response.content)

        # Asserting the incorrect answer and system's answer with result as failure
        self.assertEquals(r['result'], 'failure')