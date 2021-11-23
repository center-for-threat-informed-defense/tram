#
# TRAM -Docker Build File 
#  - This file assumes you have cloned the TRAM repo: 
#        `git@github.com:center-for-threat-informed-defense/tram.git`
#  - The working directory for this build is /path/to/tram/
#
#  Simple build command: `docker build -t [repo_name]/tram:[version] .`

FROM ubuntu:20.04

WORKDIR /tram
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip wget

COPY ./ .
RUN pip3 install -r ./requirements/requirements.txt

#COPY ./src src
#COPY ./data data
COPY ./docker/build.sh build.sh
COPY ./docker/entrypoint.sh entrypoint.sh
RUN chmod +x ./build.sh && ./build.sh

EXPOSE 8000

ENTRYPOINT [ "bash", "entrypoint.sh" ]

