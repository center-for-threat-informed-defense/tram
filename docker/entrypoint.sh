#!/bin/bash

# Prepare the TRAM server by starting the DB
python3  /mnt/tram/src/tram/manage.py makemigrations tram
python3  /mnt/tram/src/tram/manage.py migrate

# Create a super user for the Django server
# This can be set by ENV
python3 /mnt/tram/src/tram/manage.py createsuperuser --noinput

# Run the server
python3 /mnt/tram/src/tram/manage.py runserver 0.0.0.0:8000