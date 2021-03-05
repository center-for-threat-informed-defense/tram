from django.contrib import admin

from tram.models import AttackTechnique, Document, DocumentProcessingJob, Indicator, Mapping, Report


class IndicatorInline(admin.TabularInline):
    extra = 0
    model = Indicator


class MappingInline(admin.TabularInline):
    extra = 0
    model = Mapping


class ReportAdmin(admin.ModelAdmin):
    inlines = [IndicatorInline, MappingInline]


admin.site.register(AttackTechnique)
admin.site.register(Document)
admin.site.register(DocumentProcessingJob)
admin.site.register(Mapping)
admin.site.register(Report, ReportAdmin)