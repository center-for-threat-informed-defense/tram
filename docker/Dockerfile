#
# Docker Build File - TRAM
#  docker build -t repo/tram:1.0

FROM ubuntu:20.04

RUN apt-get update
RUN apt install -y python3.9
RUN apt install -y python3-pip

COPY ./requirements/requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY ./docker/entrypoint.sh entrypoint.sh

EXPOSE 8000

ENTRYPOINT [ "bash", "entrypoint.sh" ]

