[![codecov](https://codecov.io/gh/center-for-threat-informed-defense/tram/branch/master/graph/badge.svg?token=YISO1NSAMZ)](https://codecov.io/gh/center-for-threat-informed-defense/tram)

# TRAM

Threat Report ATT&CK Mapping (TRAM) is an open-source platform designed to advance research into automating the mapping of cyber threat intelligence reports to MITRE ATT&CK®.

This project aims to accelerate research into automated TTP identification in threat intel reports to greatly reduce the time and effort required to integrate new intelligence into cyber operations. TRAM enables researchers to test and refine Machine Learning (ML) models for identifying ATT&CK techniques in prose-based threat intel reports and allows threat intel analysts to train ML models and validate ML results. 

Through research into automating the mapping of cyber threat intel reports to ATT&CK, TRAM aims to reduce the cost and increase the effectiveness of integrating ATT&CK into cyber threat intelligence across the community. Threat intel providers, threat intel platforms, and analysts should be able to use TRAM to integrate ATT&CK more easily and consistently into their products.

## Table of contents
* [Installation](#installation)
* [Requirements](#requirements)
* [Developer Setup](#developer-setup)
* [Contribute](#how-do-i-contribute)
* [Notice](#notice)

## Installation
1. Get Docker: https://docs.docker.com/get-docker/
2. Download docker-compose.yml (view raw, save as)
```
https://github.com/center-for-threat-informed-defense/tram/blob/master/docker/docker-compose.yml
```
3. If desired, edit the settings in docker-compose.yml
4. Navigate to the directory where you saved docker-compose.yml

5. Run TRAM using docker
```
docker compose -f docker/docker-compose.yml up
```

6. Navigate to http://localhost:8000/ and login using the username and password specified in docker-compose.yml
![image](https://user-images.githubusercontent.com/2951827/129959436-d36e8d1f-fe74-497e-b549-a74be8d140ca.png)


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
We welcome your feedback and contributions to help advance TRAM. Please see the guidance for contributors if are you interested in [contributing or simply reporting issues.](/CONTRIBUTING.md)

Please submit [issues](https://github.com/center-for-threat-informed-defense/tram/issues) for any technical questions/concerns or contact ctid@mitre-engenuity.org directly for more general inquiries.

## Notice
Copyright 2021 MITRE Engenuity. Approved for public release. Document number CT0035

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

This project makes use of MITRE ATT&CK®

[ATT&CK Terms of Use](https://attack.mitre.org/resources/terms-of-use/)
