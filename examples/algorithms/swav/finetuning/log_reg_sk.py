import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
parser.add_argument('--load_file', required=True,
                    help='Pickle file of precomputed features')
parser.add_argument('--save_dir', default='.',
                    help='Where to save the results')
parser.add_argument('--num_reg_values', type=int, default=50,
                    help='Number of regularization values to sweep over.')

MAX_ITERS = 200

def get_acc(preds, labels):
    return np.mean(preds == labels)

def normalize_features(train, test):
    # compute mean and std based on train
    # take the mean over the entire vectorized design matrix, so that the
    # regularization values are in the same range regardless of the
    # dataset size
    mean = np.mean(train[0])
    std = np.std(train[0])
    train[0] = (train[0] - mean) / std
    test[0] = (test[0] - mean) / std

def test_log_reg_warm_starting(train, test, num_reg_values):
    Cs = np.logspace(-7, 0, num_reg_values)
    clf = LogisticRegression(random_state=0, warm_start=True, max_iter=MAX_ITERS)
    accs = []
    for C in Cs:
        clf.C = C
        # fit to train
        clf.fit(train[0], train[1])
        test_preds = clf.predict(test[0])
        test_acc = get_acc(test_preds, test[1])
        print(f'Test Accuracy', test_acc)
        accs.append(test_acc)
    return accs

def main():
    args = parser.parse_args()
    with open(args.load_file, 'rb') as f:
        train, test = pickle.load(f)
    normalize_features(train, test)
    accs = test_log_reg_warm_starting(train, test, args.num_reg_values)
    with open(os.path.join(args.save_dir, "linear_probe_eval.txt"), 'wb') as f:
        pickle.dump(accs, f)

if __name__ == "__main__":
    main()
