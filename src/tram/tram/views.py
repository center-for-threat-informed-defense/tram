import os

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

from tram.models import Document, DocumentProcessingJob

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