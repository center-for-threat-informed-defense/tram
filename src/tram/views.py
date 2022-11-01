import json
import logging
import os
import tempfile
import time
from urllib.parse import quote

import requests
from constance import config
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from rest_framework import renderers, viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from tram import serializers
from tram.ml import base
from tram.models import (
    AttackObject,
    Document,
    DocumentProcessingJob,
    Mapping,
    Report,
    Sentence,
)
from tram.renderers import DocxReportRenderer

logger = logging.getLogger(__name__)


class AttackObjectViewSet(viewsets.ModelViewSet):
    queryset = AttackObject.objects.all()
    serializer_class = serializers.AttackObjectSerializer


class DocumentProcessingJobViewSet(viewsets.ModelViewSet):
    queryset = DocumentProcessingJob.objects.all()
    serializer_class = serializers.DocumentProcessingJobSerializer


class MappingViewSet(viewsets.ModelViewSet):
    queryset = Mapping.objects.all()
    serializer_class = serializers.MappingSerializer

    def get_queryset(self):
        queryset = MappingViewSet.queryset
        sentence_id = self.request.query_params.get("sentence-id", None)
        if sentence_id:
            queryset = queryset.filter(sentence__id=sentence_id)

        return queryset


class ReportViewSet(viewsets.ModelViewSet):
    queryset = Report.objects.all()
    serializer_class = serializers.ReportSerializer


class ReportMappingViewSet(viewsets.ModelViewSet):
    """
    This viewset provides access to report mappings.
    """

    serializer_class = serializers.ReportExportSerializer
    renderer_classes = [renderers.JSONRenderer, DocxReportRenderer]

    def get_queryset(self):
        """
        Override parent implementation to support lookup by document ID.
        """
        queryset = Report.objects.all()
        document_id = self.request.query_params.get("doc-id", None)
        if document_id:
            queryset = queryset.filter(document__id=document_id)

        return queryset

    def retrieve(self, request, pk=None):
        """
        Get the mappings for a report.

        Overrides the parent implementation to add a Content-Disposition header
        so that the browser will download instead of displaying inline.

        :param request: HTTP request
        :param pk: primary key of a report
        """
        response = super().retrieve(request, request, pk)
        report = self.get_object()
        filename = "{}.{}".format(
            quote(report.name, safe=""), request.accepted_renderer.format
        )
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


class SentenceViewSet(viewsets.ModelViewSet):
    queryset = Sentence.objects.all()
    serializer_class = serializers.SentenceSerializer

    def get_queryset(self):
        queryset = SentenceViewSet.queryset
        report_id = self.request.query_params.get("report-id", None)
        if report_id:
            queryset = queryset.filter(report__id=report_id)

        attack_id = self.request.query_params.get("attack-id", None)
        if attack_id:
            sentences = Mapping.objects.filter(
                attack_object__attack_id=attack_id
            ).values("sentence")
            queryset = queryset.filter(id__in=sentences)
        return queryset


@login_required
def index(request):
    jobs = DocumentProcessingJob.objects.all()
    job_serializer = serializers.DocumentProcessingJobSerializer(jobs, many=True)

    reports = Report.objects.all()
    report_serializer = serializers.ReportSerializer(reports, many=True)

    context = {
        "job_queue": job_serializer.data,
        "reports": report_serializer.data,
    }

    return render(request, "index.html", context=context)


@login_required
@require_POST
def upload(request):
    """Places a file into ml-pipeline for analysis"""
    # Initialize the processing job.
    dpj = None

    # Initialize response.
    response = {"message": "File saved for processing."}

    file_content_type = request.FILES["file"].content_type
    if file_content_type in (
        "application/pdf",  # .pdf files
        "text/html",  # .html files
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx files
        "text/plain",  # .txt files
    ):
        dpj = DocumentProcessingJob.create_from_file(
            request.FILES["file"], request.user
        )
    elif file_content_type in ("application/json",):  # .json files
        json_data = json.loads(request.FILES["file"].read())
        res = serializers.ReportExportSerializer(data=json_data)

        if res.is_valid():
            res.save(created_by=request.user)
        else:
            return HttpResponseBadRequest(res.errors)
    else:
        return HttpResponseBadRequest("Unsupported file type")

    if dpj:
        response["job-id"] = dpj.pk
        response["doc-id"] = dpj.document.pk

    # mm = base.ModelManager("logreg") //LogisticRegressionModel
    # mm = base.ModelManager("nb") #NaiveBayesModel
    # mm = base.ModelManager("nn_cls")    # nn_cls MLPClassifierModel
    # mm.run_model()

    return JsonResponse(response)


@login_required
def ml_home(request):
    techniques = AttackObject.get_sentence_counts()
    model_metadata = base.ModelManager.get_all_model_metadata()

    context = {
        "techniques": techniques,
        "ML_ACCEPT_THRESHOLD": config.ML_ACCEPT_THRESHOLD,
        "ML_CONFIDENCE_THRESHOLD": config.ML_CONFIDENCE_THRESHOLD,
        "models": model_metadata,
    }

    return render(request, "ml_home.html", context)


@login_required
def ml_technique_sentences(request, attack_id):
    techniques = AttackObject.objects.all().order_by("attack_id")
    techniques_serializer = serializers.AttackObjectSerializer(techniques, many=True)

    context = {"attack_id": attack_id, "attack_techniques": techniques_serializer.data}
    return render(request, "technique_sentences.html", context)


@login_required
def ml_model_detail(request, model_key):
    try:
        model_metadata = base.ModelManager.get_model_metadata(model_key)
    except ValueError:
        raise Http404("Model does not exists")
    context = {"model": model_metadata}
    return render(request, "model_detail.html", context)


@login_required
def analyze(request, pk):
    report = Report.objects.get(id=pk)
    techniques = AttackObject.objects.all().order_by("attack_id")
    techniques_serializer = serializers.AttackObjectSerializer(techniques, many=True)

    context = {
        "report_id": report.id,
        "report_name": report.name,
        "attack_techniques": techniques_serializer.data,
    }
    return render(request, "analyze.html", context)


@login_required
def download_document(request, doc_id):
    """Download a verbatim copy of a previously uploaded document."""
    doc = Document.objects.get(id=doc_id)
    docfile = doc.docfile

    try:
        with docfile.open("rb") as report_file:
            response = HttpResponse(
                report_file, content_type="application/octet-stream"
            )
            filename = quote(docfile.name)
            response["Content-Disposition"] = f"attachment; filename={filename}"
    except IOError:
        raise Http404("File does not exist")

    return response


@api_view(["POST"])
def train_model(request, name):
    """
    Train the specified model.

    Runs training synchronously and returns a response when the training is
    complete.

    :param name: the name of the model
    """
    try:
        model = base.ModelManager(name)
    except ValueError:
        raise Http404("Model does not exist")

    logger.info(f"Training ML Model: {name}")
    start = time.time()
    model.train_model()
    elapsed = time.time() - start
    logger.info("Trained ML model in %0.3f seconds", elapsed)

    return Response(
        {
            "message": "Model successfully trained.",
            "elapsed_sec": elapsed,
        }
    )


def parse_generated_report(unique_identifier, report_json):
    technique_list = []
    temp = {}
    final_json = {}

    for single_sentence in report_json["sentences"]:
        for mapping_object in single_sentence["mappings"]:
            temp["technique_id"] = mapping_object["attack_id"]
            temp["confidence"] = mapping_object["confidence"]
            technique_list.append(temp)
            temp = {}

    final_json["unique_identifier"] = unique_identifier
    final_json["technique_list"] = technique_list

    return final_json


################


def filter_description(description_list):
    filter_desc = []
    for single_object in description_list:
        field_data = str(single_object["field_data"])
        if field_data[-1] != ".":
            field_data += "."
        filter_desc.append(field_data)
    return filter_desc


def create_temporary_file(description):
    tempfile.tempdir = os.getcwd() + "/data/media"
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(str.encode(description))
        fp.seek(0)
        data = fp.read()
        print(data)

        b = open(fp.name + ".txt", "wb")
        b.write(str.encode(description))
        b.close()
        filePath = fp.name + ".txt"
        fileName = str(fp.name).split("/")[-1] + ".txt"

    return fileName, filePath


@api_view(["POST"])
def techniques(request):

    session_token = request.auth.token.decode("utf-8")
    response = []

    file_content_type = request.content_type
    if file_content_type in ("application/json",):  # .json files
        json_data = request.data

        for single_object in json_data:
            control_id = single_object["unique_identifier"]

            description = " ".join(filter_description(single_object["description"]))

            filePath = ""
            fileName, filePath = create_temporary_file(description)
            uploaded_report_status = upload_text(filePath, fileName, session_token)
            uploaded_report_status = json.loads(
                uploaded_report_status.content.decode("utf-8")
            )
            generated_doc_id = uploaded_report_status["doc-id"]

            # mm = base.ModelManager("logreg") //LogisticRegressionModel
            # mm = base.ModelManager("nb") #NaiveBayesModel
            mm = base.ModelManager("nn_cls")  # nn_cls MLPClassifierModel
            mm.run_model()

            ress = extract_report(session_token)
            ress = json.loads(ress.decode("utf-8"))
            generated_report = {}
            for single_report in ress:
                document_id = single_report.get("document_id", None)
                if document_id == generated_doc_id:
                    generated_report = single_report
            final_report = {}
            if generated_report != {}:
                final_report = parse_generated_report(control_id, generated_report)

            response.append(final_report)
    else:
        return HttpResponseBadRequest("Unsupported file type")
    os.remove(filePath)
    write_logs(
        control_id, description, fileName, filePath, final_report, generated_doc_id
    )
    return JsonResponse(response, safe=False)


def extract_report(session_token):

    url = "http://localhost:8000/api/report-mappings/"

    payload = {}
    headers = {"Authorization": "Bearer " + session_token}

    response = requests.request("GET", url, headers=headers, data=payload, timeout=1000)

    return response.content


@api_view(["POST"])
def modified_upload(request):
    """Places a file into ml-pipeline for analysis"""
    # Initialize the processing job.
    dpj = None

    # Initialize response.
    response = {"message": "File saved for processing."}

    file_content_type = request.FILES["file"].content_type
    if file_content_type in (
        "application/pdf",  # .pdf files
        "text/html",  # .html files
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx files
        "text/plain",  # .txt files
    ):
        dpj = DocumentProcessingJob.create_from_file(
            request.FILES["file"], request.user
        )
    elif file_content_type in ("application/json",):  # .json files
        json_data = json.loads(request.FILES["file"].read())
        res = serializers.ReportExportSerializer(data=json_data)

        if res.is_valid():
            res.save(created_by=request.user)
        else:
            return HttpResponseBadRequest(res.errors)
    else:
        return HttpResponseBadRequest("Unsupported file type")

    if dpj:
        response["job-id"] = dpj.pk
        response["doc-id"] = dpj.document.pk
    return JsonResponse(response)


def upload_text(filePath, fileName, session_token):
    url = "http://localhost:8000/modified_upload/"

    payload = {}
    files = [
        (
            "file",
            (
                fileName,
                open(filePath, "rb"),
                "text/plain",
            ),
        )
    ]
    headers = {"Authorization": "Bearer " + session_token}
    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files, timeout=1000
    )
    return response


def write_logs(
    unique_identifier,
    description,
    file_name,
    file_path,
    processed_data,
    generated_doc_id,
):
    _log = {}
    _log["unique_identifier"] = unique_identifier
    _log["description"] = description
    _log["temporary_file_name"] = file_name
    _log["temporary_file_path"] = file_path
    _log["processed_data"] = processed_data
    _log["document_id"] = generated_doc_id

    print("----------------------------------------------------------------")
    print(_log)
    print("*****************************************************************")
