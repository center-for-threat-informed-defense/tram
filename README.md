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

## Getting Started
TRAM has integrated Machine Learning models into a Web UI and as part of Jupyter notebooks.
 
* Follow the [installation](URL) instructions in the Wiki to pull the container images. If you’ve used TRAM before, you’re familiar with launching into the webUI and uploading a JSON, docx, pdf, or even txt report to for automatic analysis. 

* Jupyter Notebooks can be found in [user_notebooks](https://github.com/center-for-threat-informed-defense/tram/tree/main/user_notebooks)  for the SciBERT-based single-label model and multi-label model. There are supplemental notebooks tailored to further fine-tune each model with additional data. Links found in that section will also open the notebooks in [Google Colab](https://colab.research.google.com), an online service that enables GPU-focused workloads.

Resource | Description
 -- | --
 [Installation Instructions]((https://github.com/center-for-threat-informed-defense/tram/wiki#installation)) | Instructions for downloading and installing TRAM container images
 [Developer Setup](https://github.com/center-for-threat-informed-defense/tram/wiki#for-developers)  | Instructions for developing TRAM. Only required for local development and testing. The containerized version is recommended for non-developers.
 [Jupyter Notebooks](https://github.com/center-for-threat-informed-defense/tram/tree/main/user_notebooks) | SciBERT-based single-label model and multi-label model notebooks. Notebooks for further fine-tuning both single and multi-label models.
 [Documentation](https://github.com/center-for-threat-informed-defense/tram/wiki) | Complete documentation for TRAM

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
