import json

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from tram.models import Document, DocumentProcessingJob


@pytest.fixture
def user():
    user = User.objects.create_superuser(username="testuser")
    # This password hash is generated for testing purposes only. The plaintext
    # is "12345". Note: iterations is set to 1 for efficiency in unit tests.
    user.password = "pbkdf2_sha256$1$SALT$r7FG7eWxROmt3/JEaZcAklA5VT9Vu8SnG9d9yeiJ72w="
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


@pytest.fixture
def document(logged_in_client):
    """Upload a document"""
    f = SimpleUploadedFile(
        "sample-document.txt", b"test file content", content_type="text/plain"
    )
    data = {"file": f}
    logged_in_client.post("/upload/", data)
    doc = Document.objects.get(docfile="sample-document.txt")
    yield doc
    doc.delete()


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
        Document.objects.get(docfile__icontains="test-report").delete()

        # Assert
        assert response.status_code == 200
        assert b"File saved for processing" in response.content
        assert doc_count_pre + 1 == doc_count_post
        assert job_count_pre + 1 == job_count_post

    def test_report_export_upload_creates_report(self, logged_in_client):
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
class TestUploadApi:
    def test_upload_api(self, logged_in_client):
        # Arrange
        f = SimpleUploadedFile(
            "test-report.pdf", b"test file content", content_type="application/pdf"
        )
        data = {"file": f}

        # Act
        response = logged_in_client.post("/api/upload/", data)

        # Assert
        assert response.status_code == 200  # HTTP 200 Ok

    def test_upload_api_404(self, logged_in_client):
        # Act
        response = logged_in_client.post("/api/upload/")

        # Assert
        assert response.status_code == 400  # HTTP 400 Bad Request


@pytest.mark.django_db
class TestMappingViewSet:
    def test_get_mappings(self, logged_in_client):
        # Act
        response = logged_in_client.get("/api/mappings/")
        json_response = json.loads(response.content)

        # Assert
        # Number of mappings in ATT&CK data fixture:
        assert len(json_response) == 163

    def test_get_mapping(self, logged_in_client, mapping):
        # Act
        response = logged_in_client.get(f"/api/mappings/{mapping.id}/")
        json_response = json.loads(response.content)

        # Assert
        assert json_response["attack_id"] == "T1059"

    def test_get_mappings_by_sentence(self, logged_in_client, mapping):
        # Act
        response = logged_in_client.get(
            f"/api/mappings/?sentence-id={mapping.sentence.id}"
        )
        json_response = json.loads(response.content)

        # Assert
        assert len(json_response) == 1
        assert json_response[0]["attack_id"] == "T1059"


@pytest.mark.django_db
class TestSentenceViewSet:
    def test_get_sentences(self, logged_in_client):
        # Act
        response = logged_in_client.get("/api/sentences/")
        json_response = json.loads(response.content)

        # Assert
        # The number of sentences in test-training-data.json:
        assert len(json_response) == 163
        assert json_response[0]["order"] == 1000

    def test_get_sentence(self, logged_in_client, sentence):
        # Act
        response = logged_in_client.get(f"/api/sentences/{sentence.id}/")
        json_response = json.loads(response.content)

        # Assert
        assert json_response["order"] == 1000

    def test_get_sentences_by_report(self, logged_in_client, sentence):
        # Act
        response = logged_in_client.get(
            f"/api/sentences/?report-id={sentence.report.id}"
        )
        json_response = json.loads(response.content)

        # Assert
        # The number of sentences in test-training-data.json:
        assert len(json_response) == 163
        assert json_response[0]["order"] == 1000

    def test_get_sentences_by_technique(self, logged_in_client):
        # Act
        response = logged_in_client.get("/api/sentences/?attack-id=T1189")
        json_response = json.loads(response.content)

        # Assert
        assert len(json_response) == 10
        assert json_response[0]["order"] == 1000


@pytest.mark.django_db
class TestReportMappings:
    def test_get_json(self, logged_in_client, mapping):
        # Act
        response = logged_in_client.get("/api/report-mappings/1/?format=json")
        json_response = json.loads(response.content)

        # Assert
        assert json_response["id"] == 1
        assert len(json_response["sentences"][0]["mappings"]) == 1

    def test_get_docx(self, logged_in_client, mapping):
        """
        Check that something that looks like a Word doc was returned.

        There are separate unit tests for the doc's content.
        """
        # Act
        response = logged_in_client.get("/api/report-mappings/1/?format=docx")
        data = response.content

        # Assert
        assert data.startswith(b"PK\x03\x04")

    def test_bootstrap_training_data_can_be_posted_as_json_report(
        self, logged_in_client
    ):
        # Arrange
        with open("tests/data/test-training-data.json") as f:
            json_string = f.read()

        # Act
        response = logged_in_client.post(
            "/api/report-mappings/", json_string, content_type="application/json"
        )

        # Assert
        assert response.status_code == 201  # HTTP 201 Created

    def test_report_export_update_not_implemented(self, logged_in_client):
        # Act
        response = logged_in_client.post(
            "/api/report-mappings/1/", "{}", content_type="application/json"
        )

        # Assert
        assert response.status_code == 405  # Method not allowed

    def test_download_original_report(self, logged_in_client, document):
        # Act
        response = logged_in_client.get(f"/api/download/{document.id}")

        # Assert
        assert response.content == b"test file content"

    def test_get_reports_by_doc_id(self, logged_in_client, report_with_document):
        # Act
        doc_id = report_with_document.document.id
        response = logged_in_client.get(f"/api/report-mappings/?doc-id={doc_id}")
        json_response = json.loads(response.content)

        # Assert
        assert json_response[0]["document_id"] == doc_id


@pytest.mark.django_db
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

    def test_train_model(self, logged_in_client):
        # Act
        response = logged_in_client.post("/api/train-model/dummy")

        # Assert
        assert response.status_code == 200  # HTTP 200 Ok

    def test_train_model_404(self, logged_in_client):
        # Act
        response = logged_in_client.post("/api/train-model/doesnt-exist")

        # Assert
        assert response.status_code == 404  # HTTP 404 Not Found
