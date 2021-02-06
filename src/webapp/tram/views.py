import os

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

@login_required
def upload(request):
    """Places a file into ml-pipeline for analysis
    """
    if request.method != 'POST':
        return HttpResponse('Request method must be POST', status=405)
    
    filename = request.FILES['file'].name
    filepath = os.path.join(settings.ML_PIPELINE_SOURCE, filename)
    if os.path.exists(filepath):
        return HttpResponse('File was not saved. A file with the same filename was previously uploaded and has not been processed.', status=200)

    import pdb
    pdb.set_trace()
    with open(filepath, 'wb') as fd:
        contents = request.FILES['file'].read()
        fd.write(contents)

    return HttpResponse('File saved for processing by ml-pipeline', status=200)