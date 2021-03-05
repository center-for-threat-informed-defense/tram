import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory containing training data
TRAINING_DATA = os.path.join(BASE_DIR, '..', 'data', 'training-data')

# Directory containing test data
TEST_DATA = os.path.join(BASE_DIR, '..', 'data', 'test-data')

# Directory reports will be read from while pipeline is running
REPORT_SOURCE = os.path.join(BASE_DIR, '..', 'data', 'reports')

# Directory where reports will be archived after being processed
REPORT_ARCHIVE = os.path.join(BASE_DIR, '..', 'data', 'archive')

# Directory where results will be created after a report is processed
RESULTS_DESTINATION = os.path.join(BASE_DIR, '..', 'data', 'results')

# Directory where models will be stored
MODEL_STORAGE = os.path.join(BASE_DIR, '..', 'data', 'ml-models')