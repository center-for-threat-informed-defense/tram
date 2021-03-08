from django.test import TestCase

from tram.ml import base


class TestIndicator(TestCase):
    def test_indicator_repr_is_correct(self):
        # Arrange
        expected = 'Indicator: MD5=54b0c58c7ce9f2a8b551351102ee0938'

        # Act
        ind = base.Indicator(type_='MD5',
                             value='54b0c58c7ce9f2a8b551351102ee0938')

        # Assert
        self.assertEqual(str(ind), expected)


class TestSentence(TestCase):
    def test_sentence_stores_no_mapping(self):
        # Arrange
        text = 'this is text'
        order = 0
        mappings = None

        # Arraange / Act
        s = base.Sentence(text, order, mappings)

        # Assert
        self.assertEqual(s.text, text)
        self.assertEqual(s.order, order)
        self.assertEqual(s.mappings, mappings)


class TestMapping(TestCase):
    def test_mapping_repr_is_correct(self):
        # Arrange
        confidence = 95.342000
        attack_technique = 'T1327'
        expected = 'Confidence=95.342000; Technique=T1327'

        # Act
        m = base.Mapping(confidence, attack_technique)

        self.assertEqual(str(m), expected)


class TestReport(TestCase):
    def test_report_stores_properties(self):
        # Arrange
        name = 'Test report'
        text = 'Test report text'
        sentences = [
            base.Sentence('test sentence text', 0, None)
        ]
        indicators = [
            base.Indicator('MD5', '54b0c58c7ce9f2a8b551351102ee0938')
        ]

        # Act
        rpt = base.Report(
            name=name,
            text=text,
            sentences=sentences,
            indicators=indicators
        )

        # Assert
        self.assertEqual(rpt.name, name)
        self.assertEqual(rpt.text, text)
        self.assertEqual(rpt.sentences, sentences)
        self.assertEqual(rpt.indicators, indicators)
