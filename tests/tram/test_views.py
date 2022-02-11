import json

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from tram.models import Document, DocumentProcessingJob


@pytest.fixture
def user():
    user = User.objects.create_superuser(username="testuser")
    user.set_password("12345")
    user.save()
    yield user
    user.delete()


@pytest.fixture
def client(user):
    client = Client()
    return client


@pytest.fixture
def logged_in_client(client):
    client.login(username="testuser", password="12345")
    return client


@pytest.mark.django_db
class TestLogin:
    def test_get_login_loads_login_form(self, client):
        # Act
        response = client.get("/login/")

        # Assert
        assert b"<title>Login</title>" in response.content

    def test_valid_login_redirects(self, client):
        # Arrange
        data = {"username": "testuser", "password": "12345"}

        # Act
        response = client.post("/login/", data)

        # Assert
        assert response.status_code == 302
        assert response.url == "/"

    def test_invalid_login_rerenders_login(self, client):
        # Arrange
        data = {"username": "not-a-real-user", "password": "password"}

        # Act
        response = client.post("/login/", data)

        # Assert
        assert response.status_code == 200
        assert b"<title>Login</title>" in response.content


@pytest.mark.django_db
class TestDocumentation:
    def test_documentation_loads(self, logged_in_client):
        # Act
        response = logged_in_client.get("/docs/")

        # Assert
        assert response.status_code == 200
        assert b"<title>Documentation</title>" in response.content


@pytest.mark.django_db
class TestIndex:
    def test_index_loads_with_no_stored_data(self, logged_in_client):
        # Act
        response = logged_in_client.get("/")

        # Assert
        assert response.status_code == 200
        assert b"<title>TRAM - Threat Report ATT&CK Mapper</title>" in response.content

    def test_index_loads_with_one_stored_report(self, logged_in_client, report):
        # Act
        response = logged_in_client.get("/")

        # Assert
        assert response.status_code == 200
        assert b"<title>TRAM - Threat Report ATT&CK Mapper</title>" in response.content

    def test_index_loads_with_one_job_queued(
        self, logged_in_client, document_processing_job
    ):
        # Act
        response = logged_in_client.get("/")

        # Assert
        assert response.status_code == 200
        assert b"<title>TRAM - Threat Report ATT&CK Mapper</title>" in response.content


@pytest.mark.django_db
class TestAnalyze:
    def test_analyze_loads(self, logged_in_client, report):
        # Act
        response = logged_in_client.get("/analyze/1/")

        # Assert
        assert response.status_code == 200
        assert b"<title>TRAM - Analyze Report</title>" in response.content


@pytest.mark.django_db
class TestUpload:
    def test_get_upload_returns_405(self, logged_in_client):
        # Act
        response = logged_in_client.get("/upload/")

        # Assert
        assert response.status_code == 405

    def test_file_upload_succeeds_and_creates_job(self, logged_in_client):
        # Arrange
        f = SimpleUploadedFile(
            "test-report.pdf", b"test file content", content_type="application/pdf"
        )
        data = {"file": f}
        doc_count_pre = Document.objects.all().count()
        job_count_pre = DocumentProcessingJob.objects.all().count()

        # Act
        response = logged_in_client.post("/upload/", data)
        doc_count_post = Document.objects.all().count()
        job_count_post = DocumentProcessingJob.objects.all().count()
        Document.objects.get(docfile="test-report.pdf").delete()

        # Assert
        assert response.status_code == 200
        assert b"File saved for processing" in response.content
        assert doc_count_pre + 1 == doc_count_post
        assert job_count_pre + 1 == job_count_post

    def test_report_export_upload_creates_report(
        self, logged_in_client, load_attack_data
    ):
        # Act
        with open("tests/data/report-for-simple-testdocx.json") as f:
            response = logged_in_client.post("/upload/", {"file": f})

        # Assert
        assert response.status_code == 200

    def test_upload_unsupported_file_type_causes_bad_request(self, logged_in_client):
        # Arrange
        f = SimpleUploadedFile(
            "test-document.zip", b"test file content", content_type="application/zip"
        )
        data = {"file": f}

        # Act
        response = logged_in_client.post("/upload/", data)

        # Assert
        assert response.status_code == 400
        assert response.content == b"Unsupported file type"


@pytest.mark.django_db
class TestMappingViewSet:
    def test_get_mappings(self, logged_in_client, mapping):
        # Act
        response = logged_in_client.get("/api/mappings/")
        json_response = json.loads(response.content)

        # Assert
        assert len(json_response) == 1
        assert json_response[0]["attack_id"] == "T1327"

    def test_get_mapping(self, logged_in_client, mapping):
        # Act
        response = logged_in_client.get("/api/mappings/1/")
        json_response = json.loads(response.content)

        # Assert
        assert json_response["attack_id"] == "T1327"

    def test_get_mappings_by_sentence(self, logged_in_client, mapping):
        # Act
        response = logged_in_client.get("/api/mappings/?sentence-id=1")
        json_response = json.loads(response.content)

        # Assert
        assert len(json_response) == 1
        assert json_response[0]["attack_id"] == "T1327"


@pytest.mark.django_db
class TestSentenceViewSet:
    def test_get_sentences(self, logged_in_client, sentence):
        # Act
        response = logged_in_client.get("/api/sentences/")
        json_response = json.loads(response.content)

        # Assert
        assert len(json_response) == 1
        assert json_response[0]["order"] == 0

    def test_get_sentence(self, logged_in_client, sentence):
        # Act
        response = logged_in_client.get("/api/sentences/1/")
        json_response = json.loads(response.content)

        # Assert
        assert json_response["order"] == 0

    def test_get_sentences_by_report(self, logged_in_client, sentence):
        # Act
        response = logged_in_client.get("/api/sentences/?report-id=1")
        json_response = json.loads(response.content)

        # Assert
        assert len(json_response) == 1
        assert json_response[0]["order"] == 0


@pytest.mark.django_db
class TestReportExport:
    def test_get_report_export_succeeds(self, logged_in_client, mapping):
        # Act
        response = logged_in_client.get("/api/report-export/1/")
        json_response = json.loads(response.content)

        # Assert
        assert "sentences" in json_response
        assert len(json_response["sentences"][0]["mappings"]) == 1

    def test_bootstrap_training_data_can_be_posted_as_json_report(
        self, logged_in_client, load_attack_data
    ):
        # Arrange
        with open("data/training/bootstrap-training-data.json") as f:
            json_string = f.read()

        # Act
        response = logged_in_client.post(
            "/api/report-export/", json_string, content_type="application/json"
        )

        # Assert
        assert response.status_code == 201  # HTTP 201 Created

    def test_report_export_update_not_implemented(self, logged_in_client):
        # Act
        response = logged_in_client.post(
            "/api/report-export/1/", "{}", content_type="application/json"
        )

        # Assert
        assert response.status_code == 405  # Method not allowed


@pytest.mark.django_db
@pytest.mark.usefixtures(
    "load_attack_data",
)
class TestMl:
    def test_ml_home_returns_http_200_ok(self, logged_in_client):
        # Act
        response = logged_in_client.get("/ml/")

        # Assert
        assert response.status_code == 200  # HTTP 200 Ok

    def test_ml_model_detail_returns_http_200_ok(self, logged_in_client):
        # Act
        response = logged_in_client.get("/ml/models/dummy")

        # Assert
        assert response.status_code == 200  # HTTP 200 Ok

    def test_ml_model_detail_returns_http_404_for_invalid_model(self, logged_in_client):
        # Act
        response = logged_in_client.get("/ml/models/this-should-not-work")

        # Assert
        assert response.status_code == 404  # HTTP 200 Ok
