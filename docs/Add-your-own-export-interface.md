Add your own Export Interface
====================

Tram has made it possible to easily add in export interfaces through some easy changes to two files, objects/c_report.py and 
static/js/basic.js.

The following steps illustrate adding a potential xml export interface that exports the display id in xml format.

## Add your type to the exports

The first step is to add the export type under the file objects/c_report.py in the EXPORTS array:

`EXPORTS = ['default', 'stix']` -> `EXPORTS = ['default', 'stix', 'xml']`

## Add code to format data

The next step is to modify the following section under objects/c_report.py:
```
def export(self, type): # what to return for each export type
        if type == 'stix':
            data = self.display
            output_stix = {}
            output_stix['objects'] = []
            output_stix['id'] = 'bundle--'+data['id']
            for sent in data['sentences']:
                if(len(sent['matches']) > 0):
                    for i in sent['matches']:
                        temp = {}
                        key = 'attack-pattern--'+i['id']
                        temp['type'] = 'attack-pattern'
                        temp['id'] = key
                        temp['name'] = i['search']['name']
                        temp['description'] = sent['text']
                        output_stix['objects'].append(temp)
            output_stix['type'] = 'bundle'
            output_stix['output_type'] = 'json'
            return output_stix
        else:
            output = self.display
            output['output_type'] = 'json'
            return output
```

add a if statement, where it checks the type by text

```
else if type == 'xml':
    # your code to format the self.display data here
    # Here's some example code that takes the id and returns it in xml format
    data = self.display
    data_str = '<xml>'
    data_str += '<id>' + data['id'] + '</id>'
    data_str += '</xml>'
    output_data = {}
    output_data['data'] = data_str
    output_data['output_type'] = 'xml'
    return output_data
```

## Add filetype export if needed

If the data needs to be exported in a format that isn't supported yet (such as xml in this example), 
you can add code to support this interface.

in the file static/js/basic.js, in the exportReport function under the downloadObject function, add
an if statement that checks for your output type:
```
else if(data['output_type'] === 'xml'){
    downloadObjectAsXml(data);
}
```
This function is also in the code as a comment.

Also add the function `downloadObjectAsXml(data)` (or one that matches your export type)
This should be another function in the `exportReport` function:

```
function exportReport(type){
    // code
    ...

    function downloadObjectAsXml(data){
        let dataStr = 'data:Application/octet-stream,' + encodeURIComponent(data['data']) 
        let downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", report_id + ".xml");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }
}
```

## Deploy export interface

With the new export type added, simply restart tram and click on the export button on any analyzed report. You
should get a dropdown of all the current export formats including your own.