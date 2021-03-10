from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase

from tram.models import Document, DocumentProcessingJob


class TestLogin(TestCase):
    def setUp(self):
        user = User.objects.create_superuser(username='testuser')
        user.set_password('12345')
        user.save()

        self.client = Client()

    def test_get_login_loads_login_form(self):
        # Act
        response = self.client.get('/login/')

        # Assert
        self.assertIn(b'Please sign in', response.content)

    def test_valid_login_redirects(self):
        # Arrange
        data = {'username': 'testuser',
                'password': '12345'}

        # Act
        response = self.client.post('/login/', data)

        # Assert
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

    def test_invalid_login_rerenders_login(self):
        # Arrange
        data = {'username': 'not-a-real-user',
                'password': 'password'}

        # Act
        response = self.client.post('/login/', data)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please sign in', response.content)


class TestUpload(TestCase):
    def setUp(self):
        user = User.objects.create_superuser(username='testuser')
        user.set_password('12345')
        user.save()

        self.client = Client()
        self.client.login(username='testuser', password='12345')

    def test_get_upload_returns_405(self):
        # Act
        response = self.client.get('/upload/')

        # Assert
        self.assertEqual(response.status_code, 405)

    def test_file_upload_succeeds_and_creates_job(self):
        # Arrange
        f = SimpleUploadedFile('test-report.pdf', b'test file content')
        data = {'file': f}
        doc_count_pre = Document.objects.all().count()
        job_count_pre = DocumentProcessingJob.objects.all().count()

        # Act
        response = self.client.post('/upload/', data)
        doc_count_post = Document.objects.all().count()
        job_count_post = DocumentProcessingJob.objects.all().count()
        Document.objects.get(docfile='test-report.pdf').delete()

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.content, b'File saved for processing')
        self.assertEqual(doc_count_pre + 1, doc_count_post)
        self.assertEqual(job_count_pre + 1, job_count_post)
