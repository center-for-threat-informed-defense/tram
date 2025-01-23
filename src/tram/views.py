import json
import logging
import time
from urllib.parse import quote

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
def upload_web(request):
    return upload(request)


@api_view(["POST"])
def upload_api(request):
    return upload(request)


def upload(request):
    """Places a file into ml-pipeline for analysis"""
    # Initialize the processing job.
    dpj = None

    # Initialize response.
    response = {"message": "File saved for processing."}

    if request.FILES.get("file") is None:
        return HttpResponseBadRequest("'file' field is required but was not provided")

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
