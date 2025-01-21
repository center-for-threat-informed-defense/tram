# syntax=docker/dockerfile:1

#
# TRAM -Docker Build File
#  - This file assumes you have cloned the TRAM repo:
#        `git@github.com:center-for-threat-informed-defense/tram.git`
#  - The working directory for this build is /path/to/tram/
#
#  Simple build command: `docker build -t [repo_name]/tram:[version] .`

FROM ubuntu:22.04

# OCI labels
LABEL "org.opencontainers.image.title"="TRAM"
LABEL "org.opencontainers.image.url"="https://ctid.mitre-engenuity.org/our-work/tram/"
LABEL "org.opencontainers.image.source"="https://github.com/center-for-threat-informed-defense/tram"
LABEL "org.opencontainers.image.description"="Threat Report ATT&CK Mapper"
LABEL "org.opencontainers.image.license"="Apache-2.0"

# Arguments
ARG TRAM_CA_URL
ARG TRAM_CA_THUMBPRINT

# directory to install nltk data
ARG nltk_data_dir="/tram/.venv/nltk_data"

# directory to put bert trained model
ARG bert_data_dir="/tram/data/ml-models/bert_model"

# Default URLs to datasets used by nltk
# NOTE: No spaces allowed around equal sign
ARG punkt_url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"
ARG wordnet_url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
ARG omw_url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip"
ARG bert_model_url="https://ctidtram.blob.core.windows.net/tram-models/single-label-202308303/pytorch_model.bin"
ARG bert_config_url="https://ctidtram.blob.core.windows.net/tram-models/single-label-202308303/config.json"

# local filenames to make dockerfile easier
ARG punkt_localfile="punkt.zip"
ARG wordnet_localfile="wordnet.zip"
ARG omw_localfile="omw.zip"
ARG bert_model_localfile="pytorch_model.bin"
ARG bert_config_localfile="config.json"

# Change default shell to bash so that we can use pipes (|) safely. See:
# https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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
    python3-wheel \
    unzip

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

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN mkdir /tram && \
    python3 -m venv /tram/.venv && \
    /tram/.venv/bin/python3 -m pip install -U pip wheel setuptools

# add venv to path
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 \
    PATH=/tram/.venv/bin:${PATH}

# flush all output immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /tram

COPY ./ .

# install app dependencies
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install -r ./requirements/requirements.txt && \
    python3 -m pip install --editable . && \
    cp -f ./docker/entrypoint.sh entrypoint.sh && \
    # Download NLTK data \
    # remove local bert model if it exists \
    rm -f ${bert_data_dir}/${bert_model_localfile} && \
    rm -f ${bert_data_dir}/${bert_config_localfile} && \
    # Download NLTK data \
    mkdir -p ${nltk_data_dir}/{corpora,tokenizers} && \
    curl -kJL -o ${nltk_data_dir}/tokenizers/${punkt_localfile} $punkt_url && \
    curl -kJL -o ${nltk_data_dir}/corpora/${omw_localfile} $omw_url && \
    curl -kJL -o ${nltk_data_dir}/corpora/${wordnet_localfile} $wordnet_url && \
    curl -kJL -o ${bert_data_dir}/${bert_model_localfile} $bert_model_url && \
    curl -kJL -o ${bert_data_dir}/${bert_config_localfile} $bert_config_url

# run this command without cache volume mounted, so model is stored on image
RUN python3 -c "import os; import transformers; os.environ['CURL_CA_BUNDLE'] = ''; mdl = transformers.AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'); mdl.save_pretrained('/tram/data/ml-models/priv-allenai-scibert-scivocab-uncased')"

# Generate and Run Django migrations scripts, collectstatic app files
RUN tram makemigrations tram && \
    tram migrate && \
    tram collectstatic

## run ml training
RUN tram attackdata load && \
    tram pipeline load-training-data && \
    tram pipeline train --model nb && \
    tram pipeline train --model logreg

EXPOSE 8000

ENTRYPOINT [ "/tram/entrypoint.sh" ]
