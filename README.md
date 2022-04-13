# TRAM

[![codecov](https://codecov.io/gh/center-for-threat-informed-defense/tram/branch/master/graph/badge.svg?token=YISO1NSAMZ)](https://codecov.io/gh/center-for-threat-informed-defense/tram)

Threat Report ATT&CK Mapping (TRAM) is an open-source platform designed to
advance research into automating the mapping of cyber threat intelligence
reports to MITRE ATT&CK®.

TRAM enables researchers to test and refine Machine Learning (ML) models for
identifying ATT&CK techniques in prose-based cyber threat intel reports and
allows threat intel analysts to train ML models and validate ML results.

Through research into automating the mapping of cyber threat intel reports to
ATT&CK, TRAM aims to reduce the cost and increase the effectiveness of
integrating ATT&CK into cyber threat intelligence across the community. Threat
intel providers, threat intel platforms, and analysts should be able to use TRAM
to integrate ATT&CK more easily and consistently into their products.

## Table of contents

- [TRAM](#tram)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Air Gap Installation](#air-gap-installation)
  - [Installation Troubleshooting](#installation-troubleshooting)
    - [\[97438\] Failed to execute script docker-compose](#97438-failed-to-execute-script-docker-compose)
  - [Report Troubleshooting](#report-troubleshooting)
    - [How long until my queued report is complete?](#how-long-until-my-queued-report-is-complete)
    - [Why is my report stuck in queued?](#why-is-my-report-stuck-in-queued)
    - [Do I have to manually accept all of the parsed sentences in the report?](#do-i-have-to-manually-accept-all-of-the-parsed-sentences-in-the-report)
  - [Requirements](#requirements)
  - [For Developers](#for-developers)
    - [Developer Setup](#developer-setup)
    - [Makefile Targets](#makefile-targets)
    - [Custom CA Certificate](#custom-ca-certificate)
  - [Machine Learning Development](#machine-learning-development)
    - [Existing ML Models](#existing-ml-models)
    - [Creating Your Own ML Model](#creating-your-own-ml-model)
  - [How do I contribute?](#how-do-i-contribute)
    - [Contribute Training Data](#contribute-training-data)
  - [Notice](#notice)

## Installation

1. Install Docker tools:
    * Docker: <https://docs.docker.com/get-docker/>
    * Docker Compose: <https://docs.docker.com/compose/install/>
    * Verify that Docker is running by running `docker ps` from a shell. If it
      shows, "CONTAINER ID   IMAGE     COMMAND" on the first line, then it is
      running. If it says, "cannot connect to Docker daemon," then Docker is not
      running.
2. Download docker-compose.yml for TRAM, using [this
   link](https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/787143e4f41f40e4aeb72d811a9d4297c03364d9/docker/docker-compose.yml)
   or using curl:

    ```shell
    $ curl -O https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/787143e4f41f40e4aeb72d811a9d4297c03364d9/docker/docker-compose.yml
    ```

3. If desired, edit the settings in `docker-compose.yml`. See
   [docker/README.md](docker/README.md) for more information.
4. Use Docker Compose to start the TRAM containers.
    * Run this command from the same directory where you downloaded
      `docker-compose.yml`.
        ```shell
        $ docker-compose up
        ```
    * The first time you run this command, it will download about 1GB of Docker
      images. This requires a connection to the internet. If your environment
      does not have a connection to the internet, refer to [Air Gap
      Installation](#air-gap-installation).
    * Once the images are downloaded, TRAM will do a bit of initialization. The following output lines indicate that TRAM is ready to use:
        ```
        tram_1   | [2022-03-30 16:18:44 +0000] [29] [INFO] Starting gunicorn 20.1.0
        tram_1   | [2022-03-30 16:18:44 +0000] [29] [INFO] Listening at: http://0.0.0.0:8000 (29)
        ```
    * _Note: the log shows the IP address 0.0.0.0, but TRAM requires connections to use one of the hostnames defined in the `ALLOWED_HOSTS` environment variable._

5. Navigate to <http://localhost:8000/> and login using the username and
   password specified in `docker-compose.yml`.
   ![image](https://user-images.githubusercontent.com/2951827/129959436-d36e8d1f-fe74-497e-b549-a74be8d140ca.png)

6. To shut down TRAM, type <kbd>Ctrl+C</kbd> in the shell where `docker-compose
   up` is running.

### Air Gap Installation

If you are unable to pull images from Docker Hub (i.e. due to corporate
firewall, airgap, etc.), it is possible to download the images and move them
onto the Docker host manually:

1. Pull the images onto a machine that is able to access Docker Hub:

    ```shell
    $ docker pull ghcr.io/center-for-threat-informed-defense/tram:latest
    $ docker pull ghcr.io/center-for-threat-informed-defense/tram-nginx:latest
    ```

2. Export the Docker images to compressed archive (`.tgz`) format:

    ```shell
    $ docker save ghcr.io/center-for-threat-informed-defense/tram:latest \
        | gzip > tram-latest.tgz
    $ docker save ghcr.io/center-for-threat-informed-defense/tram-nginx:latest \
        | gzip > tram-nginx-latest.tgz
    ```
3. Confirm that the images were exported correctly.

    ```shell
    ls -lah tram*.tgz
    -rw-r--r--  1 johndoe  wheel   345M Feb 24 12:56 tram-latest.tgz
    -rw-r--r--  1 johndoe  wheel   9.4M Feb 24 12:57 tram-nginx-latest.tgz
    ```

4. Copy the images across the airgap.
   - _This step will depend on your deployment environment, of course._

5. Import the Docker images on the Docker host.

    ```shell
    $ docker load < tram-latest.tgz
    $ docker load < tram-nginx-latest.tar.gz
    ```

6. Confirm that the images were loaded on the Docker host.

    ```shell
    $ docker images | grep tram
    ghcr.io/center-for-threat-informed-defense/tram-nginx   latest    8fa8fb7801b9   2 weeks ago    23.5MB
    ghcr.io/center-for-threat-informed-defense/tram         latest    d19b35523098   2 weeks ago    938MB
    ```

7. From this point, you can follow the main installation instructions above.

## Installation Troubleshooting

### \[97438\] Failed to execute script docker-compose

If you see this stack trace:

```shell
Traceback (most recent call last):
  File "docker-compose", line 3, in <module>
  File "compose/cli/main.py", line 81, in main
  File "compose/cli/main.py", line 200, in perform_command
  File "compose/cli/command.py", line 60, in project_from_options
  File "compose/cli/command.py", line 152, in get_project
  File "compose/cli/docker_client.py", line 41, in get_client
  File "compose/cli/docker_client.py", line 170, in docker_client
  File "docker/api/client.py", line 197, in __init__
  File "docker/api/client.py", line 221, in _retrieve_server_version
docker.errors.DockerException: Error while fetching server API version: ('Connection aborted.', ConnectionRefusedError(61, 'Connection refused'))
[97438] Failed to execute script docker-compose
```

Then most likely Docker is not running and you need to start Docker.

## Report Troubleshooting

### How long until my queued report is complete?

A queued report should only take about a minute to complete.

### Why is my report stuck in queued?

This is likely a problem with the processing pipeline. If the pipeline is not
working when you are running TRAM via docker, then this might be a TRAM-level
bug. If you think this is the case, then please file an issue and we can tell
you how to get logs off the system to troubleshoot.

### Do I have to manually accept all of the parsed sentences in the report?

Yes. The workflow of TRAM is that the AI/ML process will propose mappings, but a
human analyst needs to validate/accept the proposed mappings.

## Requirements

* [python3](https://www.python.org/) (3.7+)
* Google Chrome is our only supported/tested browser

## For Developers

### Developer Setup

The following steps are only required for local development and testing. The
containerized version is recommended for non-developers.

1. Install the following packages using your OS package manager (apt, yum, homebrew, etc.):
   1. make
   2. shellcheck
   3. shfmt

2. Start by cloning this repository.

    ```sh
    git clone git@github.com:center-for-threat-informed-defense/tram.git
    ```

3. Change to the TRAM directory.

    ```sh
    cd tram/
    ```

4. Create a virtual environment and activate the new virtual environment.
   1. Mac and Linux

      ```sh
      python3 -m venv venv
      source venv/bin/activate
      ```

   2. Windows

      ```bat
      venv\Scripts\activate.bat
      ```

5. Install Python application requirements.

    ```sh
    pip install -r requirements/requirements.txt
    pip install -r requirements/test-requirements.txt
    ```

6. Install pre-commit hooks

    ```sh
    pre-commit install
    ```

7. Set up the application database.

    ```sh
    tram makemigrations tram
    tram migrate
    ```

8. Run the Machine learning training.

    ```sh
    tram attackdata load
    tram pipeline load-training-data
    tram pipeline train --model nb
    tram pipeline train --model logreg
    tram pipeline train --model nn_cls
    ```

9.  Create a superuser (web login)

    ```sh
    tram createsuperuser
    ```

10. Run the application server

    ```sh
    DEBUG=1 tram runserver
    ```

11. Open the application in your web browser.
    1. Navigate to <http://localhost:8000> and use the superuser to log in
12. In a separate terminal window, run the ML pipeline

     ```sh
     cd tram/
     source venv/bin/activate
     tram pipeline run
     ```

### Makefile Targets

- Run TRAM application
  - `make start-container`
- Stop TRAM application
  - `make stop-container`
- View TRAM logs
  - `make container-logs`
- Build Python virtualenv
  - `make venv`
- Install production Python dependencies
  - `make install`
- Install prod and dev Python dependencies
  - `make install-dev`
- Manually run pre-commit hooks without performing a commit
  - `make pre-commit-run`
- Build container image (By default, container is tagged with timestamp and git hash of codebase) _See note below about custom CA certificates in the Docker build.)_
  - `make build-container`
- Run linting locally
  - `make lint`
- Run unit tests, safety, and bandit locally
  - `make test`

The automated test suite runs inside `tox`, which guarantees a consistent testing
environment, but also has considerable overhead. When writing code, it may be
useful to run `pytest` directly, which is considerably faster and can also be
used to run a specific test. Here are some useful pytest commands:

```shell
# Run the entire test suite:
$ pytest tests/

# Run tests in a specific file:
$ pytest tests/tram/test_models.py

# Run a test by name:
$ pytest tests/ -k test_mapping_repr_is_correct

# Run tests with code coverage tracking, and show which lines are missing coverage:
$ pytest --cov=tram --cov-report=term-missing tests/
```

### Custom CA Certificate

If you are building the container in an environment that intercepts SSL
connections, you can specify a root CA certificate to inject into the container
at build time. (This is only necessary for the TRAM application container. The
TRAM Nginx container does not make outbound connections.)

Export the following two variables in your environment.

```shell
$ export TRAM_CA_URL="http://your.domain.com/root.crt"
$ export TRAM_CA_THUMBPRINT="C7:E0:F9:69:09:A4:A3:E7:A9:76:32:5F:68:79:9A:85:FD:F9:B3:BD"
```

The first variable is a URL to a PEM certificate containing a root certificate
that you want to inject into the container. (If you use an `https` URL, then
certificate checking is disabled.) The second variable is a SHA-1 certificate
thumbprint that is used to verify that the correct certificate was downloaded.
You can obtain the thumbprint with the following OpenSSL command:

```shell
$ openssl x509 -in <your-cert.crt> -fingerprint -noout
SHA1 Fingerprint=C7:E0:F9:69:09:A4:A3:E7:A9:76:32:5F:68:79:9A:85:FD:F9:B3:BD
```

After exporting these two variables, you can run `make build-container` as usual
and the TRAM container will contain your specified root certificate.

## Machine Learning Development

All source code related to machine learning is located in TRAM
[src/tram/ml](https://github.com/center-for-threat-informed-defense/tram/tree/master/src/tram/ml).

### Existing ML Models

TRAM has four machine learning models that can be used out-of-the-box:

1. LogisticRegressionModel - Uses SKLearn's [Logistic
   Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
2. NaiveBayesModel - Uses SKLearn's [Multinomial
   NB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB).
3. Multilayer Perception - Uses SKLearn's
   [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
4. DummyModel - Uses SKLearn's [Dummy
   Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier)
   for testing purposes.

All ML models are implemented as an SKLearn Pipeline. Other types of models can be added in the future if there is a need.

### Creating Your Own ML Model

In order to write your own model, take the following steps:

1. Create a subclass of `tram.ml.base.SKLearnModel` that implements the
   `get_model` function. See existing ML Models for examples that can be copied.

    ```python
    class DummyModel(SKLearnModel):
        def get_model(self):
            # Your model goes here
            return Pipeline([
                ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
                ("clf", DummyClassifier(strategy='uniform'))
            ])
    ```

2. Add your model to the `ModelManager`
   [registry](https://github.com/center-for-threat-informed-defense/tram/blob/a4d874c66efc11559a3faeced4130f153fa12dca/src/tram/tram/ml/base.py#L309)
   1. Note: This method can be improved. Pull requests welcome!

    ```python
    class ModelManager(object):
        model_registry = {
            'dummy': DummyModel,
            'nb': NaiveBayesModel,
            'logreg': LogisticRegressionModel,
            # Your model on the line below
            'your-model': python.path.to.your.model
        }
    ```

3. You can now train your model, and the model will appear in the application
   interface.

   ```sh
   tram pipeline train --model your-model
   ```

4. If you are interested in sharing your model with the community, thank you!
   Please [open a Pull
   Request](https://github.com/center-for-threat-informed-defense/tram/pulls)
   with your model, and please include performance statistics in your Pull
   Request description.

## How do I contribute?

We welcome your feedback and contributions to help advance TRAM. Please see the
guidance for contributors if are you interested in [contributing or simply
reporting issues.](/CONTRIBUTING.md)

Please submit
[issues](https://github.com/center-for-threat-informed-defense/tram/issues) for
any technical questions/concerns or contact ctid@mitre-engenuity.org directly
for more general inquiries.

### Contribute Training Data

All training data is formatted as a report export. If you are contributing
training data, please ensure that you have the right to publicly share the
threat report. Do not contribute reports that are proprietary material of
others.

To contribute training data, please:

1. Use TRAM to perform the mapping, and ensure that all mappings are accepted
2. Use the report export feature to export the report as JSON
3. Open a pull request where the training data is added to data/training/contrib

## Notice

Copyright 2021 MITRE Engenuity. Approved for public release. Document number
CT0035.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

This project makes use of MITRE ATT&CK®

[ATT&CK Terms of Use](https://attack.mitre.org/resources/terms-of-use/)
