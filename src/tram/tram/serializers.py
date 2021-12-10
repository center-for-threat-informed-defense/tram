from django.db import transaction
from rest_framework import serializers

from tram import models as db_models


class AttackObjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = db_models.AttackObject
        fields = ['id', 'attack_id', 'name']


class DocumentProcessingJobSerializer(serializers.ModelSerializer):
    """Needs to be kept in sync with ReportSerializer for display purposes"""
    name = serializers.SerializerMethodField()
    byline = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()

    class Meta:
        model = db_models.DocumentProcessingJob
        fields = ['id', 'name', 'byline', 'status', 'message', 'created_by', 'created_on', 'updated_on']
        order = ['-created_on']

    def get_name(self, obj):
        name = obj.document.docfile.name
        return name

    def get_byline(self, obj):
        byline = '%s on %s' % (obj.created_by, obj.created_on.strftime('%Y-%M-%d %H:%M:%S UTC'))
        return byline

    def get_status(self, obj):
        if obj.status == 'queued':
            return 'Queued'
        elif obj.status == 'error':
            return 'Error'
        else:
            return 'Unknown'


class MappingSerializer(serializers.ModelSerializer):
    attack_id = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()
    confidence = serializers.DecimalField(max_digits=100, decimal_places=1)

    class Meta:
        model = db_models.Mapping
        fields = ['id', 'attack_id', 'name', 'confidence']

    def get_attack_id(self, obj):
        return obj.attack_object.attack_id

    def get_name(self, obj):
        return obj.attack_object.name

    def to_internal_value(self, data):
        """DRF's to_internal_value function only retains model fields from the input JSON. For Mappings,
        attack_id is an important field that is not on the model (it is on a related model).

        This function overrides DRF's base to_internal_value so that those mappings are retained and
        available to the is_valid() and create() methods later on.
        """
        internal_value = super().to_internal_value(data)  # Keeps model fields

        # Add necessary fields
        attack_object = db_models.AttackObject.objects.get(attack_id=data['attack_id'])
        sentence = db_models.Sentence.objects.get(id=data['sentence'])
        report = db_models.Report.objects.get(id=data['report'])

        internal_value.update({
            'report': report,
            'sentence': sentence,
            'attack_object': attack_object,
        })

        return internal_value

    def create(self, validated_data):
        mapping = db_models.Mapping.objects.create(
            report=validated_data['report'],
            sentence=validated_data['sentence'],
            attack_object=validated_data['attack_object'],
            confidence=validated_data['confidence']
        )

        return mapping


class ReportSerializer(serializers.ModelSerializer):
    byline = serializers.SerializerMethodField()
    accepted_sentences = serializers.SerializerMethodField()
    reviewing_sentences = serializers.SerializerMethodField()
    total_sentences = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()

    class Meta:
        model = db_models.Report
        fields = ['id', 'name', 'byline', 'accepted_sentences', 'reviewing_sentences', 'total_sentences',
                  'text', 'ml_model', 'created_by', 'created_on', 'updated_on', 'status']
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


class ReportExportSerializer(ReportSerializer):
    """Defines the export format for reports. Defined separately from ReportSerializer so that:
        1. ReportSerializer and ReportExportSerializer can evolve independently
        2. The export is larger than what the REST API needs
    """
    sentences = serializers.SerializerMethodField()

    class Meta(ReportSerializer.Meta):
        fields = ReportSerializer.Meta.fields + ['sentences', ]

    def get_sentences(self, obj):
        sentences = db_models.Sentence.objects.filter(report=obj)
        sentences_serializer = SentenceSerializer(sentences, many=True)
        return sentences_serializer.data

    def to_internal_value(self, data):
        """DRF's to_internal_value function only retains model fields from the input JSON. For Report Exports,
        there are many important fields that are not on the Report model. For instance sentences and mappings.

        This function overrides DRF's base to_internal_value so that those important fields are retained and
        available to the is_valid() and create() methods later on.
        """
        internal_value = super().to_internal_value(data)  # Keeps model fields

        # Add sentences
        sentence_serializers = [SentenceSerializer(data=sentence) for sentence in data.get('sentences', [])]

        internal_value.update({'sentences': sentence_serializers})
        return internal_value

    def create(self, validated_data):
        with transaction.atomic():
            report = db_models.Report.objects.create(
                            name=validated_data['name'],
                            document=None,
                            text=validated_data['text'],
                            ml_model=validated_data['ml_model'],
                            created_by=None,  # TODO: Get user from session
                        )

            for sentence in validated_data['sentences']:
                if sentence.is_valid():
                    sentence.validated_data['report'] = report
                    sentence.save()
                else:
                    # TODO: Handle this case better
                    raise Exception('Sentence validation needs to be handled better')

        return report

    def update(self, instance, validated_data):
        raise NotImplementedError()


class SentenceSerializer(serializers.ModelSerializer):
    mappings = serializers.SerializerMethodField()

    class Meta:
        model = db_models.Sentence
        fields = ['id', 'text', 'order', 'disposition', 'mappings']

    def get_mappings(self, obj):
        mappings = db_models.Mapping.objects.filter(sentence=obj)
        mappings_serializer = MappingSerializer(mappings, many=True)
        return mappings_serializer.data

    def to_internal_value(self, data):
        """DRF's to_internal_value function only retains model fields from the input JSON. For Sentences,
        mappings are an important field that is not on the model.

        This function overrides DRF's base to_internal_value so that those mappings are retained and
        available to the is_valid() and create() methods later on.
        """
        internal_value = super().to_internal_value(data)  # Keeps model fields

        # Add mappings
        mapping_serializers = [MappingSerializer(data=mapping) for mapping in data.get('mappings', [])]

        internal_value.update({'mappings': mapping_serializers})
        return internal_value

    def create(self, validated_data):
        with transaction.atomic():
            sentence = db_models.Sentence.objects.create(
                text=validated_data['text'],
                document=None,
                report=validated_data['report'],
                disposition=validated_data['disposition']
            )

            for mapping in validated_data.get('mappings', []):
                mapping.initial_data['sentence'] = sentence.id
                mapping.initial_data['report'] = validated_data['report'].id
                if mapping.is_valid():
                    mapping.save()
                else:
                    # TODO: Handle this case better
                    raise Exception('Mapping validation needs to be handled better')

        return sentence
