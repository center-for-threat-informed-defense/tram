from django.contrib import admin

from tram.models import AttackGroup, AttackTechnique, Document, DocumentProcessingJob, \
    Indicator, Report, ReportGroupMapping, ReportTechniqueMapping, \
    Sentence, SentenceGroupMapping, SentenceTechniqueMapping


class IndicatorInline(admin.TabularInline):
    extra = 0
    model = Indicator


class SentenceTechniqueMappingInline(admin.TabularInline):
    extra = 0
    model = SentenceTechniqueMapping


class SentenceInline(admin.TabularInline):
    extra = 0
    model = Sentence
    readonly_fields = ('text', 'document', 'order')


class AttackTechniqueAdmin(admin.ModelAdmin):
    readonly_fields = ('name', 'stix_id', 'attack_id', 'attack_url', 'matrix')


class AttackGroupAdmin(admin.ModelAdmin):
    readonly_fields = ('name', 'stix_id', 'attack_id', 'attack_url', 'matrix')


class DocumentAdmin(admin.ModelAdmin):
    inlines = [SentenceInline]


class ReportAdmin(admin.ModelAdmin):
    inlines = [IndicatorInline, SentenceTechniqueMappingInline]
    readonly_fields = ('document', 'text', 'ml_model')


class SentenceAdmin(admin.ModelAdmin):
    readonly_fields = ('text', 'document', 'order')


admin.site.register(AttackTechnique, AttackTechniqueAdmin)
admin.site.register(AttackGroup, AttackGroupAdmin)
admin.site.register(Document, DocumentAdmin)
admin.site.register(DocumentProcessingJob)
admin.site.register(Report, ReportAdmin)
admin.site.register(ReportGroupMapping)
admin.site.register(ReportTechniqueMapping)
admin.site.register(Sentence, SentenceAdmin)
admin.site.register(SentenceGroupMapping)
admin.site.register(SentenceTechniqueMapping)
