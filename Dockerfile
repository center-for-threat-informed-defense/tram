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

# Arguments
ARG TRAM_CA_URL
ARG TRAM_CA_THUMBPRINT

# Change default shell to bash so that we can use pipes (|) safely. See:
# https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install and update apt dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-wheel && \
    rm -fr /var/lib/apt/lists/*

# Add proxy settings. Uncomment and update these lines as needed.
#ENV proxy_host=${proxy_host:-proxy-server.my-organization.local} \
#    proxy_port=${proxy_port:-80}
#ENV http_proxy=http://${proxy_host}:${proxy_port} \
#    https_proxy=http://${proxy_host}:${proxy_port} \
#    no_proxy=localhost,127.0.0.1
#ENV HTTP_PROXY=${http_proxy} \
#    HTTPS_PROXY=${https_proxy} \
#    NO_PROXY=${no_proxy}

# Handle custom CA certificate, if specified.
RUN if test -n "${TRAM_CA_URL}" -a -n "${TRAM_CA_THUMBPRINT}" ; then \
        echo "Installing certificate authority from ${TRAM_CA_URL}" && \
        curl -sk "${TRAM_CA_URL}" -o /usr/local/share/ca-certificates/tram_ca.crt && \
        DOWNLOAD_CA_THUMBPRINT=$(openssl x509 -in /usr/local/share/ca-certificates/tram_ca.crt -fingerprint -noout | cut -d= -f2) && \
        if test "${DOWNLOAD_CA_THUMBPRINT}" = "${TRAM_CA_THUMBPRINT}"; then \
            update-ca-certificates; \
        else \
            printf "\n=====\nERROR\nExpected thumbprint: %s\nActual thumbprint:   %s\n=====\n" "${TRAM_CA_THUMBPRINT}" "${DOWNLOAD_CA_THUMBPRINT}"; \
            exit 1; \
        fi; \
    fi

RUN mkdir /tram && \
    python3 -m venv /tram/.venv && \
    /tram/.venv/bin/python3 -m pip install -U pip wheel setuptools

# add venv to path
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 \
    PATH=/tram/.venv/bin:${PATH}

# flush all output immediately
ENV PYTHONUNBUFFERED 1

WORKDIR /tram

COPY ./ .
COPY download_nltk.py .
# install app dependencies
RUN python3 -m pip install -r ./requirements/requirements.txt && \
    python3 -m pip install --editable . && \
    cp -f ./docker/entrypoint.sh entrypoint.sh && \
    # Download NLTK data
    python3 download_nltk.py

# Generate and Run Django migrations scripts, collectstatic app files
RUN tram makemigrations tram && \
    tram migrate && \
    tram collectstatic

# run ml training
RUN tram attackdata load && \
    tram pipeline load-training-data && \
    tram pipeline train --model nb && \
    tram pipeline train --model logreg

EXPOSE 8000

ENTRYPOINT [ "/tram/entrypoint.sh" ]
