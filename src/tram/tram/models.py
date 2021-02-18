from django.db import models


class Document(models.Model):
    """Store all documents that can be analyzed to create reports
    """
    docfile = models.FileField()
    created_on = models.DateTimeField(auto_now_add=True)


class DocumentProcessingJob(models.Model):
    """Queue of document processing jobs
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    created_on = models.DateTimeField(auto_now_add=True)


class Report(models.Model):
    """Store reports
    """
    pass
