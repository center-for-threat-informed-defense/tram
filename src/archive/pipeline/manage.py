import argparse
import inspect
import ntpath
import os
import shutil
import time
import json

from watchdog.observers import Observer 
from watchdog.events import FileSystemEventHandler

from models import Tram
import settings


class ReportHandler(FileSystemEventHandler):
    def __init__(self, model, results_destination, report_archive):
        self.model = model
        self.results_destination = results_destination
        self.report_archive = report_archive
        self.files_processed = 0

    def on_any_event(self, event):
        filename = ntpath.basename(event.src_path)
        report_path = event.src_path
        archive_path = os.path.join(settings.REPORT_ARCHIVE, filename)
        result_filename = filename.replace('.', '-') + '.json'
        result_path = os.path.join(settings.RESULTS_DESTINATION, result_filename)
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            print('Processing %s' % event.src_path)
            start = time.time()
            report = self.model.create_report(event.src_path) # Analyze the report
            with open(result_path, 'w') as outfile:
                outfile.write(json.dumps(report.export('json')))
                print('... Results written to %s' % result_path)
            end = time.time()
            elapsed = end - start
            self.files_processed += 1
            # TODO: Put results in result destination in correct format
            shutil.move(event.src_path, archive_path)
            print('.. Processing completed in %fs' % elapsed)


def run(args):
    """
    Watches a directory for reports.
    One file_create events:
        1. call report_handler.on_any_event()
        2. Write results to output directory
        3. Move report to archive directory
    """
    model = Tram()
    model.load_model()
    report_handler = ReportHandler(model, settings.RESULTS_DESTINATION, settings.REPORT_ARCHIVE)
    observer = Observer()
    observer.schedule(report_handler, settings.REPORT_SOURCE, recursive=False)
    observer.start()
    print('Watching %s for new reports' % settings.REPORT_SOURCE)
    try:
        while True:
            time.sleep(1)
    except:
        observer.stop()

    observer.join()
    print('Processed %d files' % report_handler.files_processed)


def load_model_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def train(args):
    """
      1. L
    """
    tram = Tram()
    tram.x_load_data('warble')
    tram.x_train()
    #model.train(args.train_dir, args.model_file)
    return


def test(args):
    """
    1. testing is done in base_model.py (def train)
    2. Look for testing model - should output f1 score
    """
    raise NotImplementedError('Test not implemented')


# TODO: Add ability to ovverride settings.py parameters via command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()
    sp_run = sp.add_parser('run', help='Runs the ML Pipeline')
    sp_run.set_defaults(func=run)

    sp_train = sp.add_parser('train', help='Trains the ML model')
    sp_train.set_defaults(func=train)
    sp_train.add_argument('--train-dir',
                          default=settings.TRAINING_DATA,
                          help='Source directory for training data. Default = %s ' % settings.TRAINING_DATA
                          )
    sp_train.add_argument('--model-file',
                          default=settings.MODEL_STORAGE,
                          help='Where to save the trained model. Defaults to %s' % settings.MODEL_STORAGE,
                         )

    sp_test = sp.add_parser('test', help='Tests the ML model for accuracy')
    sp_test.set_defaults(func=test)

    args = parser.parse_args()
    command = args.func
    command(args)