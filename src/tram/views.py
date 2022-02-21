import io
import json
import logging
from urllib.parse import quote

from constance import config
from django.contrib.auth.decorators import login_required
from django.http import (
    Http404,
    HttpResponse,
    HttpResponseBadRequest,
    StreamingHttpResponse,
)
from django.shortcuts import render
from rest_framework import viewsets

import tram.report.docx
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


class ReportExportViewSet(viewsets.ModelViewSet):
    queryset = Report.objects.all()
    serializer_class = serializers.ReportExportSerializer

    def retrieve(self, request, *args, **kwargs):

        report_format = request.GET.get("type", "")

        # If an invalid report_format is given, just default to json
        if report_format not in ["json", "docx"]:
            report_format = "json"
            logger.warning("Invalid File Type. Defaulting to JSON.")

        # Retrieve report data as json
        response = super().retrieve(request, *args, **kwargs)
        basename = quote(self.get_object().name, safe="")

        if report_format == "json":
            response["Content-Disposition"] = f'attachment; filename="{basename}.json"'

        elif report_format == "docx":
            # Uses json dictionary to create formatted document
            document = tram.report.docx.build(response.data)

            # save document info
            buffer = io.BytesIO()
            document.save(buffer)  # save your memory stream
            buffer.seek(0)  # rewind the stream

            # put them to streaming content response within docx content_type
            content_type = (
                "application/"
                "vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            response = StreamingHttpResponse(
                streaming_content=buffer,  # use the stream's content
                content_type=content_type,
            )

            response["Content-Disposition"] = f'attachment; filename="{basename}.docx"'
            response["Content-Encoding"] = "UTF-8"

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
def upload(request):
    """Places a file into ml-pipeline for analysis"""
    if request.method != "POST":
        return HttpResponse("Request method must be POST", status=405)

    file_content_type = request.FILES["file"].content_type
    if file_content_type in (
        "application/pdf",  # .pdf files
        "text/html",  # .html files
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx files
        "text/plain",  # .txt files
    ):
        DocumentProcessingJob.create_from_file(request.FILES["file"], request.user)
    elif file_content_type in ("application/json",):  # .json files
        json_data = json.loads(request.FILES["file"].read())
        res = serializers.ReportExportSerializer(data=json_data)

        if res.is_valid():
            res.save(created_by=request.user)
        else:
            return HttpResponseBadRequest(res.errors)
    else:
        return HttpResponseBadRequest("Unsupported file type")

    return HttpResponse("File saved for processing", status=200)


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
