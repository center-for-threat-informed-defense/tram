from django.db import models

DISPOSITION_CHOICES = (
    ('accept', 'Accepted'),
    ('reject', 'Rejected')
)


class AttackTechnique(models.Model):
    """Attack Techniques
    """
    name = models.CharField(max_length=200)
    stix_id = models.CharField(max_length=128, unique=True)
    attack_id = models.CharField(max_length=128, unique=True)
    attack_url = models.CharField(max_length=512)
    matrix = models.CharField(max_length=200)

    def __str__(self):
        return '(%s) %s' % (self.attack_id, self.name)


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


class Mapping(models.Model):
    """Maps sentences to Attack TTPs
    """
    report = models.ForeignKey(Report, on_delete=models.CASCADE)
    sentence = models.TextField()
    attack_technique = models.ForeignKey(AttackTechnique, on_delete=models.CASCADE, blank=True, null=True)
    confidence = models.FloatField()
    disposition = models.CharField(max_length=200, default='accept', choices=DISPOSITION_CHOICES)