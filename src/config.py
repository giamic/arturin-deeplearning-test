import os
import pickle
from datetime import datetime

DATA_FOLDER = os.path.join('..', 'Data')
TRAIN_FOLDER = os.path.join(DATA_FOLDER, 'Train')
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, 'Validation')
LEARN_FOLDER = os.path.join(DATA_FOLDER, 'Learn')
TEST_FOLDER = os.path.join(DATA_FOLDER, 'Test')
DICTIONARY_PATH = os.path.join(DATA_FOLDER, 'dict.pkl')

MODEL_FOLDER = os.path.join('..', 'models', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

with open(DICTIONARY_PATH, 'rb') as f:
    DICTIONARY = pickle.load(f)

N_WORDS = len(DICTIONARY)
TIMESTEPS = 437
