import tram.report.docx


def test_docx_build():
    """
    Test the .docx output.

    Testing the output of Word doc conversion is inherently messy, so this test
    is just checking for some very basic well-formedness.
    """
    json_report = {
        "id": 10,
        "name": "Test Report.pdf",
        "byline": "None on 2022-02-10 18:20:51 UTC",
        "accepted_sentences": 1,
        "reviewing_sentences": 1,
        "total_sentences": 2,
        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc dictum elementum risus.",
        "ml_model": "LogisticRegressionModel",
        "created_by": None,
        "created_on": "2022-02-10T18:20:51.115411Z",
        "updated_on": "2022-02-10T18:20:51.115445Z",
        "status": "Reviewing",
        "sentences": [
            {
                "id": 25368,
                "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "order": 0,
                "disposition": "accept",
                "mappings": [
                    {
                        "id": 3145,
                        "attack_id": "T1001",
                        "name": "Data Obfuscation",
                        "confidence": "100.0",
                    }
                ],
            },
            {
                "id": 25369,
                "text": "Nunc dictum \nelementum risus.",
                "order": 1,
                "disposition": "accept",
                "mappings": [],
            },
        ],
    }

    doc = tram.report.docx.build(json_report)
    assert len(doc.paragraphs) == 9
    assert doc.paragraphs[0].text == "TRAM Test Report.pdf"
    assert (
        doc.paragraphs[1].text
        == "Accepted Sentences: 1\nReviewing Sentences: 1\nTotal Sentences: 2"
    )
    assert doc.paragraphs[2].text == "Techniques Found"
    assert (
        doc.paragraphs[3].text
        == "Total Techniques: 1\nAttack Id: T1001, Name: Data Obfuscation\n"
    )
    assert doc.paragraphs[4].text == "\n"
    assert doc.paragraphs[5].text == "Matched Sentences"
    assert doc.paragraphs[6].text == "\n"
    assert doc.paragraphs[7].text == "Full Document"
    assert (
        doc.paragraphs[8].text
        == "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc dictum elementum risus."
    )
