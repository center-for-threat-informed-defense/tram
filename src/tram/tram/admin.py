from django.contrib import admin

from tram.models import AttackTechnique, Document, DocumentProcessingJob, Indicator, Mapping, Report, Sentence


class IndicatorInline(admin.TabularInline):
    extra = 0
    model = Indicator


class MappingInline(admin.TabularInline):
    extra = 0
    model = Mapping


class SentenceInline(admin.TabularInline):
    extra = 0
    model = Sentence
    readonly_fields = ('text', 'document', 'order')


class AttackTechniqueAdmin(admin.ModelAdmin):
    readonly_fields = ('name', 'stix_id', 'attack_id', 'attack_url', 'matrix')


class DocumentAdmin(admin.ModelAdmin):
    inlines = [SentenceInline]


class ReportAdmin(admin.ModelAdmin):
    inlines = [IndicatorInline, MappingInline]
    readonly_fields = ('document', 'text', 'ml_model')


class SentenceAdmin(admin.ModelAdmin):
    readonly_fields = ('text', 'document', 'order')


admin.site.register(AttackTechnique, AttackTechniqueAdmin)
admin.site.register(Document, DocumentAdmin)
admin.site.register(DocumentProcessingJob)
admin.site.register(Mapping)
admin.site.register(Report, ReportAdmin)
admin.site.register(Sentence, SentenceAdmin)
