import io

from rest_framework import renderers

import tram.report.docx


class DocxReportRenderer(renderers.BaseRenderer):
    """This custom renderer exports mappings into Word .docx format."""

    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    format = "docx"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        """
        Export report mappings into Word .docx format.

        :param data: the report mappings dict
        :param accepted_media_type: the content type negotiated by DRF
        :param renderer_context: optional additional data
        :returns bytes: .docx binary data
        """
        document = tram.report.docx.build(data)
        buffer = io.BytesIO()
        document.save(buffer)
        buffer.seek(0)
        return buffer.read()
