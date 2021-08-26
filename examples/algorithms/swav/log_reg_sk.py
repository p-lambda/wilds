import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import os


def get_max_iters(dataset, domain_task):
    if dataset == 'breeds':
        if domain_task[0] == 'entity30':
            return 100
        if domain_task[0] == 'living17':
            return 200
        raise NotImplementedError('Other Breeds tasks not supported.')
    if dataset == 'domainnet':
        return 150


def get_acc(preds, labels):
    return np.mean(preds == labels)


def subsample_features(data_dict, id_domain, train_frac):
    train_data_dict = data_dict[id_domain]['train']
    num_samples = len(train_data_dict[0])  # id domain, train split, features
    idx = np.random.choice(num_samples, size=int(train_frac * num_samples), replace=False)
    data_dict[id_domain]['train'] = [train_data_dict[0][idx], train_data_dict[1][idx]]
    subsample_size = len(data_dict[id_domain]['train'][0])
    print(f'Subsampled to {subsample_size} features for training.')


def normalize_features(data_dict, id_domain, all_domains):
    # compute mean and std-dev based on id_domain
    train_features = data_dict[id_domain]['train'][0]
    mean = np.mean(train_features)
    stddev = np.std(train_features)
    for domain in all_domains:
        new_feat = (data_dict[domain]['train'][0] - mean) / stddev
        data_dict[domain]['train'] = [new_feat, data_dict[domain]['train'][1]]


def test_log_reg_warm_starting(data_dict, previous_args, args):
    max_iters = get_max_iters(previous_args.dataset, args.id_domain)
    Cs = np.logspace(-7, 0, args.num_reg_values)
    clf = LogisticRegression(random_state=0, warm_start=True, max_iter=max_iters)
    id_accs = []
    ood_accs = {key: [] for key in args.ood_domains}
    for C in Cs:
        clf.C = C
        # fit to id
        id_train_feat = data_dict[args.id_domain]['train'][0]
        id_train_labels = data_dict[args.id_domain]['train'][1]
        clf.fit(id_train_feat, id_train_labels)
        # get id acc
        id_preds = clf.predict(data_dict[args.id_domain]['test'][0])
        id_acc = get_acc(id_preds, data_dict[args.id_domain]['test'][1])
        print(f'ID accuracy ({args.id_domain})', id_acc)
        id_accs.append(id_acc)
        # get all ood accs
        for ood_domain in args.ood_domains:
            ood_preds = clf.predict(data_dict[ood_domain]['test'][0])
            ood_acc = get_acc(ood_preds, data_dict[ood_domain]['test'][1])
            print(f'OOD Accuracy ({ood_domain})', ood_acc)
            ood_accs[ood_domain].append(ood_acc)
    return id_accs, ood_accs


def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
    parser.add_argument('--run_dir', type=str,
                        help='Outer run directory to use.', required=True)
    parser.add_argument('--train_data_fracs', type=float, nargs='+', default=[1.0],
                        help='The amount of source data to actually use for fine-tuning.')
    parser.add_argument('--num_reg_values', type=int, default=50,
                        help='Number of regularization values to sweep over.', required=False)
    parser.add_argument('--id_domain', type=str, required=True,
                        help='The source domain on which to train')
    parser.add_argument('--ood_domains', type=str, required=True,
                        help='The target domains on which to evaluate (comma-separated).')
    parser.add_argument('--file_name', type=str, required=True,
                        help='Name of the pickle file (without directories, without .pickle).')
    parser.add_argument('--overwrite', action='store_true',
                        help='If set, will overwrite the existing files.')
    parser.add_argument('--is_breeds', action='store_true', help='Set if is breeds task.')
    args = parser.parse_args()

    if args.file_name.endswith('.pickle'):
        args.file_name = args.file_name[:-len('.pickle')]
    load_path = os.path.join(args.run_dir, 'finetuning', f'{args.file_name}.pickle')

    args.ood_domain_str = args.ood_domains
    args.ood_domains = args.ood_domains.split(',')
    if args.is_breeds:  # use source for ID, target for OOD
        args.id_domain = (args.id_domain, True)
        args.ood_domains = [(d, False) for d in args.ood_domains]

    # check that feature extraction done
    if not os.path.exists(load_path):
        raise ValueError(f'Must run extract_features_new_fmt.py first to get {load_path}. Exiting...')
    data, _ = pickle.load(open(load_path, 'rb'))
    args.all_domains = [args.id_domain] + args.ood_domains
    for domain in args.all_domains:
        if domain not in data.keys():
            raise ValueError(f'Features for {domain} have not been extracted yet. Exiting...')

    for train_data_frac in args.train_data_fracs:
        data, previous_args = pickle.load(open(load_path, 'rb'))
        save_file_name = f'lin_probe_{args.file_name}_{args.id_domain}_{args.ood_domain_str}_{train_data_frac}.pickle'
        save_path = os.path.join(args.run_dir, 'finetuning', save_file_name)
        if (not args.overwrite) and (os.path.exists(save_path)):
            print(f'Already exists results at {save_path}. Skipping...')
            continue
        print(f'Using representations from {previous_args.dataset}, source {args.id_domain}, '
              f'targets {args.ood_domain_str}, using ckpt epoch {previous_args.ckpt_epoch} from {args.run_dir}, '
              f'now using a training data fraction {train_data_frac}.')
        subsample_features(data, args.id_domain, train_data_frac)
        normalize_features(data, args.id_domain, args.all_domains)
        id_accs, ood_accs = test_log_reg_warm_starting(data, previous_args, args)
        with open(save_path, 'wb') as f:
            pickle.dump((id_accs, ood_accs), f)


if __name__ == "__main__":
    main()