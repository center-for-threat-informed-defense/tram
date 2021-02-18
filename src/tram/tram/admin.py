from django.contrib import admin

from tram.models import Document, DocumentProcessingJob

admin.site.register(Document)
admin.site.register(DocumentProcessingJob)