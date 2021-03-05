Getting started
====================

With TRAM installed and running, you can now analyze threat intelligence reports. There are two ways of doing this:

1. Enter a URL to a blog or other website. 
2. Upload a file using the upload icon. Files are stored in the data/reports directory.

In either case, TRAM will create a report object with a unique identifier (random UUID). It will then scrape the 
contents from the location (or read the file) and run it through all of the machine learning models on the backend, 
to extract ATT&CK data, as well as Indicators of Compromise (IOC).

Reports will show up grey when they are initially created, showing them as queued. They change to the todo column 
when completed (i.e., all models have been run). By clicking on the report view button, you can open the object, allowing you
to view or export the extracted data. By clicking on the report edit button, you can annotate and view the report content.
Reports can be moved to the review column, where they will turn yellow, and the completed column, where they will turn green.

Models can be retrained via the retraining dropdown. You also have the option of adding rss feeds to tram via the hamburger
dropdown.
