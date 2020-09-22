import pdfplumber

def parse_file(f):
    pieces = f.split('.')
    if not len(pieces) > 1:  # file path + extension
        return _parse_text(f)
    if pieces[-1] == 'txt':
        return _parse_text(f)
    elif pieces[-1] == 'pdf':
        return _parse_pdf(f)


def _parse_text(f):
    try:
        with open(f, 'r') as f:
            return f.read()
    except Exception as e:
        print(e)

def _parse_pdf(f):
    try:
        with pdfplumber.open(f) as pdf  :
            return ''.join(page.extract_text() for page in pdf.pages)
    except Exception as e:
        print(e)