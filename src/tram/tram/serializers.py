from rest_framework import serializers

from tram import models as db_models


class AttackTechniqueSerializer(serializers.ModelSerializer):
    class Meta:
        model = db_models.AttackTechnique
        fields = ['id', 'attack_id', 'name']


class DocumentProcessingJobSerializer(serializers.ModelSerializer):
    """Needs to be kept in sync with ReportSerializer for display purposes"""
    name = serializers.SerializerMethodField()
    byline = serializers.SerializerMethodField()
    accepted_sentences = serializers.SerializerMethodField()
    reviewing_sentences = serializers.SerializerMethodField()
    total_sentences = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()

    class Meta:
        model = db_models.DocumentProcessingJob
        fields = ['id', 'name', 'byline', 'accepted_sentences', 'reviewing_sentences', 'total_sentences',
                  'created_by', 'created_on', 'updated_on', 'status']
        order = ['-created_on']

    def get_name(self, obj):
        name = obj.document.docfile.name
        return name

    def get_byline(self, obj):
        byline = '%s on %s' % (obj.created_by, obj.created_on.strftime('%Y-%M-%d %H:%M:%S UTC'))
        return byline

    def get_accepted_sentences(self, obj):
        return 0

    def get_reviewing_sentences(self, obj):
        return 0

    def get_total_sentences(self, obj):
        return 0

    def get_status(self, obj):
        return 'Queued'


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
    accepted_sentences = serializers.SerializerMethodField()
    reviewing_sentences = serializers.SerializerMethodField()
    total_sentences = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()

    class Meta:
        model = db_models.Report
        fields = ['id', 'name', 'byline', 'accepted_sentences', 'reviewing_sentences', 'total_sentences',
                  'created_by', 'created_on', 'updated_on', 'status']
        order = ['-created_on']

    def get_accepted_sentences(self, obj):
        count = db_models.Sentence.objects.filter(disposition='accept', report=obj).count()
        return count

    def get_reviewing_sentences(self, obj):
        count = db_models.Sentence.objects.filter(disposition=None, report=obj).count()
        return count

    def get_total_sentences(self, obj):
        count = db_models.Sentence.objects.filter(report=obj).count()
        return count

    def get_byline(self, obj):
        byline = '%s on %s' % (obj.created_by, obj.created_on.strftime('%Y-%m-%d %H:%M:%S UTC'))
        return byline

    def get_status(self, obj):
        reviewing_sentences = self.get_reviewing_sentences(obj)
        status = 'Reviewing'
        if reviewing_sentences == 0:
            status = 'Accepted'
        return status


class SentenceSerializer(serializers.ModelSerializer):
    mappings = serializers.SerializerMethodField()

    class Meta:
        model = db_models.Sentence
        fields = ['id', 'text', 'order', 'disposition', 'mappings']

    def get_mappings(self, obj):
        mappings = db_models.Mapping.objects.filter(sentence=obj)
        mappings_serializer = MappingSerializer(mappings, many=True)
        return mappings_serializer.data
