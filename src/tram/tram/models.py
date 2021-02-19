from django.db import models

DISPOSITION_CHOICES = (
    ('accept', 'Accepted'),
    ('reject', 'Rejected')
)

class Document(models.Model):
    """Store all documents that can be analyzed to create reports
    """
    docfile = models.FileField()
    created_on = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.docfile.name


class DocumentProcessingJob(models.Model):
    """Queue of document processing jobs
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    created_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return 'Process %s' % self.document.docfile.name

class Report(models.Model):
    """Store reports
    """
    name = models.CharField(max_length=200)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    ml_model = models.CharField(max_length=200)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Indicator(models.Model):
    """Indicators extracted from a document for a report
    """
    report = models.ForeignKey(Report, on_delete=models.CASCADE)
    indicator_type = models.CharField(max_length=200)
    value = models.CharField(max_length=200)

    def __str__(self):
        return '%s: %s' % (self.indicator_type, self.value)


class Sentence(models.Model):
    """Sentences extracted from a document for a report
    """
    report = models.ForeignKey(Report, on_delete=models.CASCADE)
    text = models.TextField()
    # TODO: Techniques/Matches - store in table + relation; or JSON array?
    disposition = models.CharField(max_length=200, default='accept', choices=DISPOSITION_CHOICES)

    def __str__(self):
        return self.text
