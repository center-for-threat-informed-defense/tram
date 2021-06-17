import os

from django.contrib.auth.models import User
from django.core.files import File
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch.dispatcher import receiver

DISPOSITION_CHOICES = (
    ('accept', 'Accepted'),
    ('reject', 'Rejected'),
    (None, 'No Disposition'),
)

SENTENCE_PREVIEW_CHARS = 40


class AttackTechnique(models.Model):
    """Attack Techniques
    """
    name = models.CharField(max_length=200)
    stix_id = models.CharField(max_length=128, unique=True)
    attack_id = models.CharField(max_length=128, unique=True)
    attack_url = models.CharField(max_length=512)
    matrix = models.CharField(max_length=200)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    sentences = models.ManyToManyField('Sentence', through='Mapping')

    def __str__(self):
        return '(%s) %s' % (self.attack_id, self.name)


class Document(models.Model):
    """Store all documents that can be analyzed to create reports
    """
    docfile = models.FileField()
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)

    def __str__(self):
        return self.docfile.name


class DocumentProcessingJob(models.Model):
    """Queue of document processing jobs
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    created_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    @classmethod
    def create_from_file(cls, f):
        assert isinstance(f, File)
        doc = Document(docfile=f)
        doc.save()
        dpj = DocumentProcessingJob(document=doc)
        dpj.save()
        return dpj

    def __str__(self):
        return 'Process %s' % self.document.docfile.name


class Report(models.Model):
    """Store reports
    """
    name = models.CharField(max_length=200)
    document = models.ForeignKey(Document, null=True, on_delete=models.CASCADE)
    text = models.TextField()
    ml_model = models.CharField(max_length=200)
    created_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
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
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    def __str__(self):
        return '%s: %s' % (self.indicator_type, self.value)


class Sentence(models.Model):
    text = models.TextField()
    document = models.ForeignKey(Document, null=True, on_delete=models.CASCADE)
    order = models.IntegerField(default=1000)  # Sentences with lower numbers are displayed first
    report = models.ForeignKey(Report, on_delete=models.CASCADE)
    disposition = models.CharField(max_length=200, default=None, null=True, blank=True, choices=DISPOSITION_CHOICES)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    def __str__(self):
        append = ''
        if len(self.text) > SENTENCE_PREVIEW_CHARS:
            append = '...'
        return self.text[:SENTENCE_PREVIEW_CHARS] + append


class Mapping(models.Model):
    """Maps sentences to Attack TTPs
    """
    report = models.ForeignKey(Report, on_delete=models.CASCADE)
    sentence = models.ForeignKey(Sentence, on_delete=models.CASCADE)
    attack_technique = models.ForeignKey(AttackTechnique, on_delete=models.CASCADE, blank=True, null=True)
    confidence = models.FloatField()
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    def __str__(self):
        return 'Sentence "%s" to %s' % (self.sentence, self.attack_technique)


class MLSettings(models.Model):
    """Settings for Machine Learning models
    """



def _delete_file(path):
    # Deletes file from filesystem
    if os.path.isfile(path):
        os.remove(path)


@receiver(post_delete, sender=Document)
def delete_file_post_delete(sender, instance, *args, **kwargs):
    if instance.docfile:
        _delete_file(instance.docfile.path)
