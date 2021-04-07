from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
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
    reviewing_reports = []
    accepted_reports = []

    jobs = DocumentProcessingJob.objects.all()
    job_serializer = serializers.DocumentProcessingJobSerializer(jobs, many=True)

    reports = Report.objects.all()
    report_serializer = serializers.ReportSerializer(reports, many=True)

    for report in report_serializer.data:  # TODO: Implement this as an annotation in the query and not in python
        if report.get('status') == 'Accepted':
            accepted_reports.append(report)
        else:
            reviewing_reports.append(report)

    all_reports = []
    all_reports.extend(reviewing_reports)
    all_reports.extend(accepted_reports)

    context = {
        'job_queue': job_serializer.data,
        'reports': all_reports,
    }

    return render(request, 'index.html', context=context)


@login_required
def upload(request):
    """Places a file into ml-pipeline for analysis
    """
    if request.method != 'POST':
        return HttpResponse('Request method must be POST', status=405)

    DocumentProcessingJob.create_from_file(request.FILES['file'])

    return HttpResponse('File saved for processing', status=200)


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
