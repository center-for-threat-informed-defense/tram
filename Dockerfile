# syntax=docker/dockerfile:1

#
# TRAM -Docker Build File
#  - This file assumes you have cloned the TRAM repo:
#        `git@github.com:center-for-threat-informed-defense/tram.git`
#  - The working directory for this build is /path/to/tram/
#
#  Simple build command: `docker build -t [repo_name]/tram:[version] .`

FROM ubuntu:20.04

# OCI labels
LABEL "org.opencontainers.image.title"="TRAM"
LABEL "org.opencontainers.image.url"="https://ctid.mitre-engenuity.org/our-work/tram/"
LABEL "org.opencontainers.image.source"="https://github.com/center-for-threat-informed-defense/tram"
LABEL "org.opencontainers.image.description"="Threat Report ATT&CK Mapper"
LABEL "org.opencontainers.image.license"="Apache-2.0"

# Install and update apt dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-venv \
        python3-wheel

RUN mkdir /tram && \
    python3 -m venv /tram/.venv && \
    /tram/.venv/bin/python3 -m pip install -U pip wheel setuptools

# add venv to path
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 \
    PATH=/tram/.venv/bin:${PATH}

# flush all output immediately
ENV PYTHONUNBUFFERED 1

# extend python path to tram import path
ENV PYTHONPATH=/tram/src/tram:${PYTHONPATH}

WORKDIR /tram

#COPY ./src src
#COPY ./data data
COPY ./ .

# install app dependencies
RUN  --mount=type=cache,target=/root/.cache \
    python3 -m pip install -r ./requirements/requirements.txt && \
    cp -f ./docker/entrypoint.sh entrypoint.sh && \
    # Download NLTK data
    python3 -m nltk.downloader punkt && \
    python3 -m nltk.downloader wordnet && \
    python3 -m nltk.downloader omw-1.4

# Generate and Run Django migrations scripts, collectstatic app files
RUN python3 /tram/src/tram/manage.py makemigrations tram && \
    python3 /tram/src/tram/manage.py migrate && \
    python3 /tram/src/tram/manage.py collectstatic

# run ml training
RUN python3 /tram/src/tram/manage.py attackdata load && \
    python3 /tram/src/tram/manage.py pipeline load-training-data && \
    python3 /tram/src/tram/manage.py pipeline train --model nb && \
    python3 /tram/src/tram/manage.py pipeline train --model logreg

EXPOSE 8000

ENTRYPOINT [ "/tram/entrypoint.sh" ]
