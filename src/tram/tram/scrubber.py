import re

# Replace files with their respective token
def replace_extension(text):
    extensionList = ["3G2",
                "3GP",
                "7Z",
                "AI",
                "AIFF",
                "AIF",
                "APK",
                "ARJ",
                "ASPX",
                "AU",
                "AVI",
                "BAK",
                "BAT",
                "BIN",
                "BMP",
                "C",
                "CAB",
                "CDA",
                "CFG",
                "CGI",
                "PL",
                "CER",
                "CFM",
                "CLASS",
                "CGI",
                "JAVA",
                "CUR",
                "CPP",
                "CSS",
                "CSV",
                "CVS",
                "DAT",
                "DB",
                "DBF",
                "DEB",
                "DBF",
                "DIF",
                "DLL",
                "DMG",
                "DMP",
                "DOC",
                "DOCX",
                "DRV",
                "EMAIL",
                "EML",
                "EPS",
                "EXE",
                "FLV",
                "FM3",
                "FNT",
                "FON",
                "GADGET",
                "GIF",
                "H",
                "H264",
                "HQX",
                "HTM",
                "HTML",
                "ICNS",
                "ICO",
                "INI",
                "ISO",
                "JAR",
                "JPG",
                "JPEG",
                "JS",
                "JSP",
                "KEY",
                "LNK",
                "LOG",
                "M4V",
                "MAC",
                "MAP",
                "MDB",
                "MID",
                "MIDI",
                "MOV",
                "MP3",
                "MP4",
                "MOV",
                "MPG",
                "MPEG",
                "QT",
                "MSG",
                "MSI",
                "MTB",
                "MTW",
                "ODP",
                "ODT",
                "OSD",
                "OFT",
                "OGG",
                "OST",
                "OTF",
                "PART",
                "PDF",
                "PKG",
                "P65",
                "T65",
                "PHP",
                "PNG",
                "PPTX",
                "PSD",
                "PSP",
                "PY",
                "QXD",
                "RA",
                "RAR",
                "RM",
                "RTF",
                "RPM",
                "RSS",
                "SAV",
                "SH",
                "SCR",
                "SIT",
                "SQL",
                "SWIFT",
                "SWF",
                "SVG",
                "SYS",
                "TAR",
                "TEX",
                "TIF",
                "TMP",
                "TOAST",
                "TXT",
                "VB",
                "VCD",
                "VCF",
                "WAV",
                "WK3",
                "WKS",
                "WMA",
                "WMV",
                "WPD",
                "WP5",
                "XLS",
                "XLSX",
                "XHTML",
                "XML",
                "WPL",
                "WSF",
                "Z",
                "ZIP"]

    separator = "|"
    extensionRegex = separator.join(extensionList)
    pattern = "(?P<filename>[\w-]+)?\.(?P<extension>(" + extensionRegex + "))(?P<endchar>(\W|$))";
    regexPattern = re.compile(pattern, re.IGNORECASE)

    # Extracts scrubbed files
    files = [m.group().strip() for m in regexPattern.finditer(text)]

    final = regexPattern.sub(r'[FILE]\g<endchar>', text)

    return final, files

# Removes various forms of markup
def replace_markup(text):

    # Examples of markup commonly in reports-- add more to list if needed
    markupStaticList = ["\(A\)",
                    "\(B\)",
                    "\(C\)",
                    "\(D\)",
                    "\(E\)",
                    "INTELLIGENCE PURPOSES ONLY: ",
                    "Distribution limited to licensed entities: ABC, MSNB, CNN, FOX, USAT"]
    markupRangeList = ["\(A//",
                "\(B//",
                "\(C//",
                "\(D//",
                "\(E//"]

    separator = "|"
    markupStaticRegex = separator.join(markupStaticList)
    markupRangeRegex = separator.join(markupRangeList)

    pattern = "((" + markupStaticRegex + ")|((" + markupRangeRegex + ")[^)]*\)))"
    regexPattern = re.compile(pattern)

    final = regexPattern.sub("", text)
    
    return final

# Replace ips with their respective token
# Extracts ips into a dictionary
def replace_ip(text):
    ipv4 = "(?P<start>\W|^)(?P<n1>\d{1,3})\.(?P<n2>\d{1,3})\.(?P<n3>\d{1,3})\.(?P<n4>\d{1,3})(?P<ending>(:\d{1,3}|[/#]\d{1,5}))?"; 
    mac = "(?P<start>\W|^)(?P<n1>[\da-f]{2})\:(?P<n2>[\da-f]{2})\:(?P<n3>[\da-f]{2})\:(?P<n4>[\da-f]{2})\:(?P<n5>[\da-f]{2})\:(?P<n6>[\da-f]{2})";  
    ipv6 = "(?P<start>\W|^)([\da-f]{1,4}::?){3,7}([\da-f]{1,4})?(?P<ending>(:\d{1,3}|[/#\.]\d{1,5}))?";  

    regexPattern = re.compile(ipv4, re.IGNORECASE)
    ipv4list = [m.group() for m in regexPattern.finditer(text)]
    textipv4 = regexPattern.sub("\g<start>[IPv4]", text)

    regexPattern = re.compile(mac, re.IGNORECASE)
    maclist = [m.group() for m in regexPattern.finditer(text)]
    textmac = regexPattern.sub("\g<start>[MAC]", textipv4)

    regexPattern = re.compile(ipv6, re.IGNORECASE)
    ipv6list = [m.group() for m in regexPattern.finditer(text)]
    final = regexPattern.sub("\g<start>[IPv6]", textmac)

    ips = {'ipv4': ipv4list, 'mac': maclist, 'ipv6': ipv6list}

    return final, ips

# Replace urls with their respective token
def replace_url(text):

    regexPattern = re.compile("(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,7}(:[0-9]{1,5})?(\/[^\s]*)?")
    
    # Extracts scrubbed urls
    urls = [m.group() for m in regexPattern.finditer(text)]

    final = regexPattern.sub("[URL]", text)
    
    return final, urls

def replace_email(text):
     regexPattern = re.compile("[\w\.]+@[a-z]+\.[a-z]+", re.IGNORECASE)

     # Extracts scrubbed emails
     emails = [m.group() for m in regexPattern.finditer(text)]

     final = regexPattern.sub("[EMAIL]", text)

     return final, emails

# Handles various other common sources of periods
def replace_other(text):

    # Localhost
    regexPattern = re.compile("localhost\.localdomain", re.IGNORECASE)
    final = regexPattern.sub("[LOCALHOST]",text)

    # Chinese chars
    regexPattern = re.compile("[\u4e00-\u9fa5]+")
    final = regexPattern.sub("[CHINESE]", final)

    # Numbers, useful if dealing with line by line but maybe unneccessary 
    #regexPattern = re.compile("[\d,]+\.[\d]*")
    #final = regexPattern.sub("[NUMBER]", final)

    # Common periods in reports for urls
    regexPattern = re.compile("(\[\.\])|(\[ \. \])")
    final = regexPattern.sub(".", final)

    # Change hxxp to http for url extraction
    regexPattern = re.compile("hxxp", re.IGNORECASE)
    final = regexPattern.sub("http", final)

    return final

def scrub(text):

    # This order of replacing is important: url extraction is much more aggressive,
    # so start with more specific sources of periods and then slowly get more general
    scrubbedText = replace_other(text)
    scrubbedText, emails = replace_email(scrubbedText)
    scrubbedText, ips = replace_ip(scrubbedText)
    scrubbedText = replace_markup(scrubbedText)
    scrubbedText, files = replace_extension(scrubbedText)
    scrubbedText, urls = replace_url(scrubbedText)

    extractedValues = {'ipv4': ips['ipv4'], 'ipv6':ips['ipv6'], 'mac':ips['mac'], 'urls':urls, 'emails':emails, 'files':files}
    return scrubbedText, extractedValues
