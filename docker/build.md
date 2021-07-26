## What is TRAM?

Threat Report ATT&CK® Mapping (TRAM) is a tool that leverages Natural Language Processing to aid analysts in mapping finished reports to ATT&CK.

There is no shortage of cyber threat intelligence (CTI) reporting, and analysts often find themselves overburdened by the constant stream of reports. Analyzing these reports can be strenuous and tedious for analysts, often taking up large amounts of their time. Automating CTI mapping to ATT&CK will reduce analyst fatigue and improve consistency of threat intelligence mappings.

TRAM seeks to help analysts by automatically extracting adversary behaviors, which can help with the acceleration of the analysis process to prevent a backlog. With faster analysis, CTI teams can more easily operationalize their intel. While TRAM cannot replace a human analyst, it certainly can help by providing analysts with some starting data about the report.

TRAM uses natural language processing and classification techniques to extract adversary behaviors (ATT&CK techniques) from raw text which comes in the form of published threat reports. The current practice to extract these techniques relies entirely on manual analysis performed by human analysts.
This introduces problems like human error, dependence on physical availability, and demand for an extensive understanding of ATT&CK. With automation, this project will increase the quality and completeness of the ATT&CK knowledge base while reducing demand on human analysts.

## How to use this image

### Starting an instance of TRAM

Example `docker-compose.yml`,

```yaml
version: '3.5'
services:
  tram:
    image: tram
    ports:
      - "8000:8000"
    environment:
      - DATA_DIRECTORY=/data
      - SECRET_KEY=Ij0WGee73k9OESwqddmSKCx6SY9aJ_7bDojs485Z6ec # your secret key here
      - DEBUG=True
      - DJANGO_SUPERUSER_USERNAME=djangoSuperuser
      - DJANGO_SUPERUSER_PASSWORD=LEGITPassword1234 # your password here
      - DJANGO_SUPERUSER_EMAIL=LEGITSuperUser1234 # your email address here
    volumes:
      - tram-data:/data
volumes:
  tram-data:
```

### Connecting to TRAM from a browser

By default, one can connect to the container running this image at `http://localhost:8000`

### Environment Variables

`SECRET_KEY`

Is a cryptographic secrecy used by Django. This secret can be generated using the following command:

```bash
$: python3 -c "import secrets; print(secrets.token_urlsafe())"
```

`DATA_DIRECTORY`

Any ML data and DB data is stored at the path indicated at this environment variable

`DEBUG`

This is the debug setting for Django and is either `True` or `False`

`DJANGO_SUPERUSER_USERNAME`

Sets the Django super user

`DJANGO_SUPERUSER_PASSWORD`

Sets the password for the Django super user

`DJANGO_SUPERUSER_EMAIL`

Sets the email address for the Django super user
