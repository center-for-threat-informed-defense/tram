[![codecov](https://codecov.io/gh/center-for-threat-informed-defense/tram/branch/master/graph/badge.svg?token=YISO1NSAMZ)](https://codecov.io/gh/center-for-threat-informed-defense/tram)

# TRAM

Threat Report ATT&CK<sup>®</sup> Mapping (TRAM) is a tool that leverages Natural Language Processing to aid analysts in mapping finished reports to ATT&CK. 

There is no shortage of cyber threat intelligence (CTI) reporting, and analysts often find themselves overburdened by the constant stream of reports. Analyzing these reports can be strenuous and tedious for analysts, often taking up large amounts of their time. Automating CTI mapping to ATT&CK will reduce analyst fatigue and improve consistency of threat intelligence mappings. 

TRAM seeks to help analysts by automatically extracting adversary behaviors, which can help with the acceleration of the analysis process to prevent a backlog.  With faster analysis, CTI teams can more easily operationalize their intel. While TRAM cannot replace a human analyst, it certainly can help by providing analysts with some starting data about the report.

TRAM uses natural language processing and classification techniques to extract adversary behaviors (ATT&CK techniques) from raw text which comes in the form of published threat reports. The current practice to extract these techniques relies entirely on manual analysis performed by human analysts. This introduces problems like human error, dependence on physical availability, and demand for an extensive understanding of ATT&CK. With automation, this project will increase the quality and completeness of the ATT&CK knowledge base while reducing demand on human analysts. 

## Table of contents
* [Installation](#installation)
* [Requirements](#requirements)
* [Installation](#developer-setup)
* [Documentation](#documentation)
* [Machine Learning](ML.md)
* [Contribute](#how-do-i-contribute)
* [Notice](#notice)

## Installation
1. Get Docker: https://docs.docker.com/get-docker/
2. Download docker-compose.yml (view raw, save as) OR
```
https://github.com/center-for-threat-informed-defense/tram/blob/master/docker/docker-compose.yml
```
use wget
```
wget https://github.com/center-for-threat-informed-defense/tram/blob/master/docker/docker-compose.yml
```
3. If desired, edit the settings in docker/docker-compose.yml

4. Run TRAM using docker
```
docker compose -f docker/docker-compose.yml up
```

## Requirements
- [python3](https://www.python.org/) (3.7+)
- Google Chrome is our only supported/tested browser

## Developer Setup
Start by cloning this repository.
```
git clone git@github.com:center-for-threat-informed-defense/tram.git
```
Change to the TRAM directory
```
cd tram/
```
Create a virtual environment
```
virtualenv venv
```
and activate it
```
source venv/bin/activate
```
Or for Windows
```
venv\Scripts\activate.bat
```
install requirements
```
pip install -r requirements/requirements.txt
```
set up the database
```
python src/tram/manage.py makemigrations tram
python src/tram/manage.py migrate
```
create a superuser (web login)
```
python src/tram/manage.py createsuperuser
```
Run the webserver
```
python src/tram/manage.py runserver 
```
Then you can navigate to http://localhost:8000 and use the superuser to log in

In a separate terminal window, run the ML pipeline
```
cd tram/
source venv/bin/activate
python src/tram/manage.py pipeline run
```

## How do I contribute?

We welcome all the help we can get in making TRAM a more useful tool for the community. 
We have made a working prototype and acknowledge that there will need to be increased efforts in the future to maintain 
and improve it.
If you have any issues with TRAM, you can create an issue in the issues tab, we'll try to respond as soon as possible.

Read [CONTRIBUTING.md](CONTRIBUTING.md) to better understand what we're looking for. 
There's also a Developer Certificate of Origin that you'll need to sign off on.
​
## Notice

Copyright 2021 The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Case Number 19-3429.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This project makes use of ATT&CK®

ATT&CK® Terms of Use - https://attack.mitre.org/resources/terms-of-use/
