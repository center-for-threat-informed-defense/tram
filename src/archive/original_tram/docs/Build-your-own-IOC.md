Build your own IOC
====================

You can easily build your IOC parser within TRAM by adding a single Python module into the app/indicators directory.

## Write your IOC 

Each indicator should follow this format:
```bash
class Indicator:

    def __init__(self):
        self.name = 'email'

    async def find(self, report, blob):
        pass
```

An indicator should:
* Exist inside a class called Indicator
* Inside the class, the __init__ function should contain (at least) an indicator name. 
* Contain a find function, which takes a report object and a text blob. The report object is an instance of the
created report, which will have the properties outlined in the app.objects.c_report module. A blob is the actual textual 
report (either the contents from a URL or a file upload).

Additionally, inside the find function, you should append matches as you extract them:
```bash
async def find(self, report, blob):
    for ip in re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', blob):
        if self._is_valid_ip(ip):
            report.matches.append(Match(search=Search(tag='ioc', name=self.name, description=ip)))
```

The above example is from the ipv4 indicator, which simply looks for ip addresses inside a report. 
We use a regex word IP match to extract the indicator from the report. If a match
is found, we append the match to the report, applying a new app.objects.c_match object. 

## Deploy your IOC

With your indicator written, simply restart TRAM and analyze a new report. You will see your indicator results automatically
collected and displayed within the report. 