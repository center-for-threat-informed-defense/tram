# TRAM

[![MITRE ATT&CK® v13](https://img.shields.io/badge/MITRE%20ATT%26CK®-v13-red)](https://attack.mitre.org/versions/v13/)
![Build](https://img.shields.io/github/actions/workflow/status/center-for-threat-informed-defense/tram/test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/center-for-threat-informed-defense/tram?token=ejCIZhBRGr)](https://codecov.io/gh/center-for-threat-informed-defense/tram)

Threat Report ATT&CK Mapper (TRAM) is an open-source platform designed to to reduce cost
and increase the effectiveness of integrating ATT&CK  across the CTI community. It does
this by automating the mapping of cyber threat intelligence (CTI) reports to MITRE
ATT&CK®. Threat intel providers, threat intel platforms, and analysts can use TRAM to
integrate ATT&CK more easily and consistently into their products.

The platform works out of the box to identify up to 50 common ATT&CK techniques in text
documents; it also supports tailoring the model by annotating additional items and
rebuilding the model. This Wiki describes the results of the Center for Threat-Informed
Defense (CTID) research into automated ATT&CK mapping and provides details and
instructions for tailoring the platform to your organization's unique dataset.

**Table Of Contents:**

- [Getting Started](#getting-started)
- [Getting Involved](#getting-involved)
- [Questions and Feedback](#questions-and-feedback)
- [How do I contribute?](#how-do-i-contribute)
- [Notice](#notice)

## Getting Started

The TRAM web application can be deployed in a containerized environment with Docker or
Kubernetes. You should read the installation instructions to make sure that you are
comfortable with the prerequisites. Alternatively, if you want to focus on Machine
Learning Engineering, you can run the project notebooks for fine tuning your own models.

| Resource                                                                                         | Description                                                                                          |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| [Documentation](https://github.com/center-for-threat-informed-defense/tram/wiki)                 | Complete documentation for TRAM                                                                      |
| [Installation](https://github.com/center-for-threat-informed-defense/tram/wiki/Installation)     | Instructions for downloading and installing TRAM container images                                    |
| [Notebooks](https://github.com/center-for-threat-informed-defense/tram/tree/main/user_notebooks) | Jupyter notebooks for SciBERT-based single-label and multi-label models.                             |
| [Developer Setup](https://github.com/center-for-threat-informed-defense/tram/wiki/Developers)    | Instructions for contributing code changes to TRAM. Only required for local development and testing. |

## Getting Involved

There are several ways that you can get involved with this project and help advance
threat-informed defense:

- **Install the TRAM web application and try processing CTI reports.** We welcome your
  feedback on the effectiveness of using machine learning to identify TTPs in
  human-readable text.
- **Share your use cases.** We are interested in developing additional tools and
  resources to help the community understand and make threat-informed decisions in their
  risk management programs. If you have ideas or suggestions, we consider them as we
  explore additional research projects.
- **Label your own data and use the notebooks to fine tune your own models.** This is a
  complex undertaking, but it allows you to adapt TRAM to your own environment and data.
  If you have high end GPUs in your environment, you can run these notebooks on your own
  instrastructure; otherwise you can run them on the paid or free tiers of [Google
  Colab](https://colab.research.google.com/).

## Questions and Feedback

Please submit issues for any technical questions/concerns or contact
ctid@mitre-engenuity.org directly for more general inquiries.

Also see the guidance for contributors if are you interested in contributing or simply
reporting issues.

## How do I contribute?

We welcome your feedback and contributions to help advance TRAM. Please see the
guidance for contributors if are you interested in [contributing or simply
reporting issues.](/CONTRIBUTING.md)

To contribute training data, see [the Data Annotation wiki](https://github.com/center-for-threat-informed-defense/tram-private/wiki/Data-Annotation).

Please submit
[issues](https://github.com/center-for-threat-informed-defense/tram/issues) for
any technical questions/concerns or contact ctid@mitre-engenuity.org directly
for more general inquiries.

## Notice

Copyright 2021, 2023 MITRE Engenuity. Approved for public release. Document number
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
