#!/usr/bin/env bash

# Create a super user for the Django server
#   This should be set by ENV and must include:
#      - DJANGO_SUPERUSER_USERNAME
#      - DJANGO_SUPERUSER_PASSWORD
#      - DJANGO_SUPERUSER_EMAIL

# Set SKIP_CREATE_SUPERUSER to 1 to skip creating the super user on every run

# check for required environment variables
SKIP_CREATE_SUPERUSER=${SKIP_CREATE_SUPERUSER:-0}

# only check for required variables if we are not skipping the creation of the superuser
if [ "${SKIP_CREATE_SUPERUSER}" -eq 0 ]; then
    if [[ -z "${DJANGO_SUPERUSER_USERNAME:+x}" ]]; then
        echo "Environment variable DJANGO_SUPERUSER_USERNAME is not set. Skipping superuser creation."
        SKIP_CREATE_SUPERUSER=1
    fi
    if [[ -z "${DJANGO_SUPERUSER_PASSWORD:+x}" ]]; then
        echo "Environment variable DJANGO_SUPERUSER_PASSWORD is not set. Skipping superuser creation."
        SKIP_CREATE_SUPERUSER=1
    fi
    if [[ -z "${DJANGO_SUPERUSER_EMAIL:+x}" ]]; then
        echo "Environment variable DJANGO_SUPERUSER_EMAIL is not set. Skipping superuser creation."
        SKIP_CREATE_SUPERUSER=1
    fi
fi

# Generate and Run Django migrations scripts
python3 /tram/src/tram/manage.py makemigrations tram
python3 /tram/src/tram/manage.py migrate

# Used provided superuser credentials to create a superuser
if [[ "${SKIP_CREATE_SUPERUSER}" -eq 0 ]]; then

    PY_CREATE_SU_SCRIPT="
from django.contrib.auth.models import User;

username = '$DJANGO_SUPERUSER_USERNAME';
password = '$DJANGO_SUPERUSER_PASSWORD';
email = '$DJANGO_SUPERUSER_EMAIL';

if not User.objects.filter(username = username).exists():
    User.objects.create_superuser(username, email, password);
    print('Superuser created.');
else:
    print('Superuser creation skipped, user already exists.');
"
    printf "%s" "${PY_CREATE_SU_SCRIPT}" | python3 /tram/src/tram/manage.py shell
fi

nohup python3 /tram/src/tram/manage.py pipeline run --model logreg &

# Run Django on port 8000
# python3 /tram/src/tram/manage.py runserver 0.0.0.0:8000

gunicorn tram.wsgi:application -b 0.0.0.0:8000
