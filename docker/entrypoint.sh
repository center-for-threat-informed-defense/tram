#!/bin/bash

# Run Django migrations scripts
python3  /tram/src/tram/manage.py makemigrations tram
python3  /tram/src/tram/manage.py migrate

# Create a super user for the Django server
#   This should be set by ENV and must include:
#      - DJANGO_SUPERUSER_USERNAME
#      - DJANGO_SUPERUSER_PASSWORD
#      - DJANGO_SUPERUSER_EMAIL
python3 /tram/src/tram/manage.py createsuperuser --noinput

# Run the server on Loopback using port 8000
python3 /tram/src/tram/manage.py runserver 0.0.0.0:8000