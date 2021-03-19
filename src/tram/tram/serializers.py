from rest_framework import serializers

from tram import models as db_models


class AttackTechniqueSerializer(serializers.ModelSerializer):
    class Meta:
        model = db_models.AttackTechnique
        fields = ['id', 'attack_id', 'name']


class DocumentProcessingJobSerializer(serializers.ModelSerializer):
    filename = serializers.SerializerMethodField()
    byline = serializers.SerializerMethodField()

    class Meta:
        model = db_models.DocumentProcessingJob
        fields = ['id', 'filename', 'byline']

    def get_filename(self, obj):
        filename = obj.document.docfile.name
        return filename

    def get_byline(self, obj):
        byline = 'TBD-user on %s' % obj.created_on.strftime('%Y-%M-%d %H:%M:%S%z')
        return byline


class MappingSerializer(serializers.ModelSerializer):
    attack_id = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()
    confidence = serializers.DecimalField(max_digits=100, decimal_places=1)

    class Meta:
        model = db_models.Mapping
        fields = ['id', 'attack_id', 'report', 'sentence', 'name', 'confidence', 'attack_technique']

    def get_attack_id(self, obj):
        return obj.attack_technique.attack_id

    def get_name(self, obj):
        return obj.attack_technique.name


class ReportSerializer(serializers.ModelSerializer):
    byline = serializers.SerializerMethodField()
    confirmed_sentences = serializers.SerializerMethodField()
    pending_sentences = serializers.SerializerMethodField()

    class Meta:
        model = db_models.Report
        fields = ['id', 'name', 'byline', 'confirmed_sentences', 'pending_sentences']

    def get_confirmed_sentences(self, obj):
        count = db_models.Sentence.objects.filter(disposition='accept', report=obj).count()
        return count

    def get_pending_sentences(self, obj):
        count = db_models.Sentence.objects.filter(disposition=None, report=obj).count()
        return count

    def get_byline(self, obj):
        byline = 'TBD-user on %s' % obj.created_on.strftime('%Y-%M-%d %H:%M:%S%z')
        return byline


class SentenceSerializer(serializers.ModelSerializer):
    mappings = serializers.SerializerMethodField()

    class Meta:
        model = db_models.Sentence
        fields = ['id', 'text', 'order', 'disposition', 'mappings']

    def get_mappings(self, obj):
        mappings = db_models.Mapping.objects.filter(sentence=obj)
        mappings_serializer = MappingSerializer(mappings, many=True)
        return mappings_serializer.data
