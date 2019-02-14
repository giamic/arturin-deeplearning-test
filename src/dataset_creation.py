import os
import pickle
import numpy as np

from config import LEARN_FOLDER, TRAIN_FOLDER, VALIDATION_FOLDER


def train_validation_split():
    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(VALIDATION_FOLDER, exist_ok=True)

    with open(os.path.join(LEARN_FOLDER, 'sequences.pkl'), 'rb') as f:
        x_lrn = pickle.load(f)
    with open(os.path.join(LEARN_FOLDER, 'sentences.pkl'), 'rb') as f:
        s_lrn = pickle.load(f)
    with open(os.path.join(LEARN_FOLDER, 'labels.pkl'), 'rb') as f:
        y_lrn = pickle.load(f)

    N = len(x_lrn)
    np.random.seed(18)
    order = np.random.permutation(N)
    x_lrn = np.array([x_lrn[i] for i in order])
    s_lrn = np.array([s_lrn[i] for i in order])
    y_lrn = np.array([y_lrn[i] for i in order])

    msk = np.random.random(N) < 0.85
    x_trn = x_lrn[msk].tolist()
    x_vld = x_lrn[~msk].tolist()

    s_trn = s_lrn[msk].tolist()
    s_vld = s_lrn[~msk].tolist()

    y_trn = y_lrn[msk].tolist()
    y_vld = y_lrn[~msk].tolist()

    with open(os.path.join(TRAIN_FOLDER, 'sequences.pkl'), 'wb') as f:
        pickle.dump(x_trn, f)
    with open(os.path.join(TRAIN_FOLDER, 'sentences.pkl'), 'wb') as f:
        pickle.dump(s_trn, f)
    with open(os.path.join(TRAIN_FOLDER, 'labels.pkl'), 'wb') as f:
        pickle.dump(y_trn, f)
    with open(os.path.join(VALIDATION_FOLDER, 'sequences.pkl'), 'wb') as f:
        pickle.dump(x_vld, f)
    with open(os.path.join(VALIDATION_FOLDER, 'sentences.pkl'), 'wb') as f:
        pickle.dump(s_vld, f)
    with open(os.path.join(VALIDATION_FOLDER, 'labels.pkl'), 'wb') as f:
        pickle.dump(y_vld, f)
    return


if __name__ == '__main__':
    train_validation_split()
