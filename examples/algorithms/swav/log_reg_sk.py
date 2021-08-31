import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import os

MAX_ITERS = 200

def get_acc(preds, labels):
    return np.mean(preds == labels)

def normalize_features(source, target):
    # compute mean and std based on source train
    source_train_feat = source[0][0]
    mean = np.mean(source_train_feat, axis=0)
    std = np.mean(source_train_feat, axis=0)
    for i in range(2):
        source[i][0] = (source[i][0] - mean) / std
        target[i][0] = (target[i][0] - mean) / std

def test_log_reg_warm_starting(source, target, num_reg_values):
    Cs = np.logspace(-7, 0, num_reg_values)
    clf = LogisticRegression(random_state=0, warm_start=True, max_iter=MAX_ITERS)
    id_accs = []
    ood_accs = []
    for C in Cs:
        clf.C = C
        # fit to source train
        clf.fit(source[0][0], source[0][1])
        # get source test acc
        source_preds = clf.predict(source[1][0])
        source_acc = get_acc(source_preds, source[1][1])
        print(f'Source accuracy', source_acc)
        id_accs.append(source_acc)
        # get all ood accs
        target_preds = clf.predict(target[1][0])
        target_acc = get_acc(target_preds, target[1][1])
        print(f'Target Accuracy', target_acc)
        ood_accs.append(target_acc)
    return id_accs, ood_accs

def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
    parser.add_argument('--source_feat_path', required=True,
                        help='Pickle file of features of train domain')
    parser.add_argument('--target_feat_path', required=True,
                        help='Pickle file of features of test domain')
    parser.add_argument('--save_path', required=True,
                        help='Where to save the results')
    parser.add_argument('--num_reg_values', type=int, default=50,
                        help='Number of regularization values to sweep over.')
    args = parser.parse_args()

    source_feat = pickle.load(open(args.source_feat_path, 'rb'))
    target_feat = pickle.load(open(args.target_feat_path, 'rb'))
    normalize_features(source_feat, target_feat)
    accs = test_log_reg_warm_starting(source_feat, target_feat, args.num_reg_values)

    with open(args.save_path, 'wb') as f:
        pickle.dump(accs, f)

if __name__ == "__main__":
    main()
