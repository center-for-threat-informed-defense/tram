from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework import generics, permissions, viewsets

from tram.models import AttackTechnique, Document, DocumentProcessingJob, Mapping, Report, Sentence
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

    in_progress = []
    done = []
    for report in reports:
        pending_sentence_count = Sentence.objects.filter(disposition=None, report=report).count()
        if pending_sentence_count > 0:
            in_progress.append(report)
        else:
            done.append(report)

    in_progress_serializer = serializers.ReportSerializer(in_progress, many=True)
    done_serializer = serializers.ReportSerializer(done, many=True)

    context = {
        'queued_jobs': job_serializer.data,
        'needs_review': in_progress_serializer.data,
        'done': done_serializer.data,
    }

    return render(request, 'index.html', context=context)


@login_required
def upload(request):
    """Places a file into ml-pipeline for analysis
    """
    if request.method != 'POST':
        return HttpResponse('Request method must be POST', status=405)

    doc = Document(docfile=request.FILES['file'])
    doc.save()

    dpq = DocumentProcessingJob(document=doc)
    dpq.save()

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
