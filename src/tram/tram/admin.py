from django.contrib import admin

from tram.models import Document, DocumentProcessingJob, Indicator, Report, Sentence


class IndicatorInline(admin.TabularInline):
    extra = 0
    model = Indicator


class SentenceInline(admin.TabularInline):
    extra = 0
    model = Sentence


class ReportAdmin(admin.ModelAdmin):
    inlines = [IndicatorInline, SentenceInline]

admin.site.register(Document)
admin.site.register(DocumentProcessingJob)
admin.site.register(Report, ReportAdmin)