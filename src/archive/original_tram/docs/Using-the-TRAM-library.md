Using the TRAM Library
====================

While TRAM already helps with the automation of processing CTI, some users might want to take it another step. TRAM's default
user interface makes it easy to submit single reports for analysis, leaving the manual process of having to submit every single report. 
The TRAM library helps with this issue by making it possible to submit numerous reports with only a few extra lines of code.

**NOTE - Since TRAM is built on asyncio, grabbing the even loop to execute function calls is necessary**

There are three main functions that exist within the TRAM library:
1. create_report
2. get_reports
3. export_report

Each function simply does what is says. Below are example scripts for using the TRAM library.

URL List
```bash

import asyncio
import json

from tram import Tram

# Create a TRAM instance
tram = Tram()
loop = asyncio.get_event_loop()

# If analyzing locally saved reports, rather than passing a list of URLs, pass in a list of file paths of the documents
urls = ['https://www.welivesecurity.com/2019/05/22/journey-zebrocy-land/', 'https://www.fireeye.com/blog/threat-research/2020/04/apt32-targeting-chinese-government-in-covid-19-related-espionage.html']

for url in urls:
    # each report requires a name and url, if you want to specify a file, you can replace url with: file='path/to/file'
    loop.run_until_complete(tram.create_report(name=url, url=url))

# Get all reports
reports = loop.run_until_complete(tram.get_reports())

# Save all reports
for report in reports:
    data = loop.run_until_complete(tram.export_report(report['id']))
    with open('{}.json'.format(report['id']), 'w') as f:
        json.dump(data, f)
```




