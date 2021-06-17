import json

from constance import config
from django.db.models import Count, Q
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.utils.text import slugify
from rest_framework import viewsets

from tram.models import AttackTechnique, DocumentProcessingJob, Mapping, Report, Sentence
from tram import serializers


class AttackTechniqueViewSet(viewsets.ModelViewSet):
    queryset = AttackTechnique.objects.all()
    serializer_class = serializers.AttackTechniqueSerializer


class DocumentProcessingJobViewSet(viewsets.ModelViewSet):
    queryset = DocumentProcessingJob.objects.all()
    serializer_class = serializers.DocumentProcessingJobSerializer


class MappingViewSet(viewsets.ModelViewSet):
    queryset = Mapping.objects.all()
    serializer_class = serializers.MappingSerializer

    def get_queryset(self):
        queryset = MappingViewSet.queryset
        sentence_id = self.request.query_params.get('sentence-id', None)
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
        response = super().retrieve(request, *args, **kwargs)
        filename = slugify(self.get_object().name) + '.json'
        response['Content-Disposition'] = 'attachment; filename="%s"' % filename
        return response


class SentenceViewSet(viewsets.ModelViewSet):
    queryset = Sentence.objects.all()
    serializer_class = serializers.SentenceSerializer

    def get_queryset(self):
        queryset = SentenceViewSet.queryset
        report_id = self.request.query_params.get('report-id', None)
        if report_id:
            queryset = queryset.filter(report__id=report_id)

        return queryset


@login_required
def index(request):
    jobs = DocumentProcessingJob.objects.all()
    job_serializer = serializers.DocumentProcessingJobSerializer(jobs, many=True)

    reports = Report.objects.all()
    report_serializer = serializers.ReportSerializer(reports, many=True)

    context = {
        'job_queue': job_serializer.data,
        'reports': report_serializer.data,
    }

    return render(request, 'index.html', context=context)


@login_required
def upload(request):
    """Places a file into ml-pipeline for analysis
    """
    if request.method != 'POST':
        return HttpResponse('Request method must be POST', status=405)

    file_content_type = request.FILES['file'].content_type
    if file_content_type in ('application/pdf',  # .pdf files
                             'text/html',  # .html files
                             'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx files
                             ):
        DocumentProcessingJob.create_from_file(request.FILES['file'])
    elif file_content_type in ('application/json', ):  # .json files
        json_data = json.loads(request.FILES['file'].read())
        res = serializers.ReportExportSerializer(data=json_data)

        if res.is_valid():
            res.save()
        else:
            return HttpResponseBadRequest(res.errors)
    else:
        return HttpResponseBadRequest('Unsupported file type')

    return HttpResponse('File saved for processing', status=200)


@login_required
def ml_home(request):
    techniques = AttackTechnique.objects.annotate(sentence_count=
                                                  Count('sentences', filter=Q(sentences__disposition='accept'))).\
                                                  order_by('-sentence_count', 'attack_id')

    context = {
               'techniques': techniques,
               'ML_ACCEPT_THRESHOLD': config.ML_ACCEPT_THRESHOLD
               }

    return render(request, 'ml_home.html', context)


@login_required
def analyze(request, pk):
    report = Report.objects.get(id=pk)
    techniques = AttackTechnique.objects.all().order_by('attack_id')
    tecniques_serializer = serializers.AttackTechniqueSerializer(techniques, many=True)

    context = {
        'report_id': report.id,
        'report_name': report.name,
        'attack_techniques': tecniques_serializer.data,
        }
    return render(request, 'analyze.html', context)
