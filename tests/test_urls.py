from django.test import SimpleTestCase
from django.urls import reverse, resolve
from vqa.views import *

class TestUrls(SimpleTestCase):

    def test_index_url(self):
        url = reverse('index')
        print(resolve(url))
        self.assertEquals(resolve(url).func, index)

    def test_upload_url(self):
        url = reverse('upload')
        print(resolve(url))
        self.assertEquals(resolve(url).func, upload)

    def test_captcha_url(self):
        url = reverse('captcha')
        print(resolve(url))
        self.assertEquals(resolve(url).func, captcha)

    def test_captcha_api_url(self):
        url = reverse('captcha_api')
        print(resolve(url))
        self.assertEquals(resolve(url).func, captcha_api)