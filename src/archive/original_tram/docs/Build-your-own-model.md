Build your own model
====================

You can easily build your own machine learning model within TRAM by adding a single Python module into the 
app/models directory.

## Write your model 

Each model should follow this format:
```bash
class Model(BaseService):

    def __init__(self):
        self.name = 'my_model'

    async def learn(self, report, tokens):
        search = await self.get_service('data_svc').locate('search', dict(tag='ATT&CK'))
        report.completed_models += 1
```

A model should:
* Exist inside a class called model, which extends the TRAM BaseService class (which has a few helper
functions for models). 
* Inside the class, the __init__ function should contain (at least) a model name. 
* Contain a learn function, which takes a report object and a list of tokens. The report object is an instance of the
created report, which will have the properties outlined in the app.objects.c_report module. Tokens are a list of each
sentence from the report (either the contents from a URL or a file upload).
* Inside the learn function, get a list of all ATT&CK data (view the app.objects.c_search for the available properties).
You can use the objects in this list to conduct your analysis. 
* At the end of the learn function, we should increment the report completed_models property, which let's it know
we have completed the model analysis.

Additionally, inside the learn function, you should append matches as you extract them:
```bash
async def learn(self, report, tokens):
    search = await self.get_service('data_svc').locate('search', dict(tag='attack'))
    for sentence in tokens:
        try:
            for s in search:
                for _ in re.findall(r'\b%s\b' % s.code, sentence):
                    report.matches.append(Match(model=self.name, search=s, confidence=100))
        except Exception as e:
            print(e)
    report.completed_models += 1
```

The above example is from the regex model, which simply looks for exact matches for techniques in a given report. 
Looping through each sentence (token), we use a regex word match to extract techniques from the report. If a match
is found, we append the match to the matches property on the report, applying a new app.objects.secondclass.c_match
object. 

Each c_match instance should contain:

* The name of the model that it was found
* The search object that we found (i.e., ATT&CK data)
* The confidence (0-100) we have that the match is valid

## Deploy your model

With your model written, simply restart TRAM and analyze a new report. You will see your model results automatically
collected and displayed within the report. 