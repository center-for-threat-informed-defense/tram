#!/bin/bash
cd /home/ubuntu/tram && source venv/bin/activate && DJANGO_DEBUG=1 tram runserver 0:8000
