import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from pathlib import Path
from PIL import Image
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import MulticlassDetectionAccuracy
from wilds.datasets.wilds_dataset import WILDSDataset


class BDD100KDataset(WILDSDataset):
    """
    Common base class for the BDD100K-wilds detection and classification datasets.
    See docstrings of classes below for details specific to each dataset.

    Input (x):
        1280x720 RGB images of driving scenes from dashboard POV.

    Website:
        https://bdd-data.berkeley.edu/

    Original publication:
        @InProceedings{bdd100k,
            author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
                      Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
            title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
            booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            month = {June},
            year = {2020}
        }

    License (original text):
        Copyright ©2018. The Regents of the University of California (Regents). All Rights Reserved.
        Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and
        not-for-profit purposes, without fee and without a signed licensing agreement; and permission use, copy, modify and
        distribute this software for commercial purposes (such rights not subject to transfer) to BDD member and its affiliates,
        is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in
        all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck
        Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu,
        http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
        IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
        INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED
        OF THE POSSIBILITY OF SUCH DAMAGE.
        REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
        AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED
        "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
    """

    _dataset_name = 'bdd100k'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x0ac62ae89a644676a57fa61d6aa2f87d/contents/blob/',
            'compressed_size': None,
        },
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        super(BDD100KDataset, self).__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        img = Image.open(self.root / 'images' / self._image_array[idx])
        return img


class BDD100KDetDataset(BDD100KDataset):
    """
    The BDD100K-wilds detection driving dataset.
    This is a modified version of the original BDD100K dataset.

    Supported `split_scheme`: 'official'

    Metadata:
        Each data point is annotated with a time of day, either 'daytime' or 'nondaytime'

    TODO: fill in more details, revisit once the rest comes together
    """
    CATEGORIES = {'bicycle': 1, 'car': 2, 'pedestrian': 3,
                  'traffic light': 4, 'traffic sign': 5}
    GROUPS = defaultdict(lambda: 1, {'daytime': 0})

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (1280, 720)
        self.root = Path(self.data_dir)

        if not split_scheme == 'official':
            raise ValueError("For BDD100K-wilds detection, split scheme should be 'official'.")
        self._metadata_fields = ['timeofday']
        self._split_scheme = split_scheme
        self._is_detection = True
        self._is_classification = False
        self._y_size = None
        self._n_classes = 5  #TODO

        with open(self.root / 'train_data.json', 'r') as f:
            train_data = json.load(f)
        with open(self.root / 'val_data.json', 'r') as f:
            val_data = json.load(f)
        with open(self.root / 'test_data.json', 'r') as f:
            test_data = json.load(f)
        self._image_array = []
        self._split_array, self._y_array, self._metadata_array = [], [], []

        for i, data in enumerate([train_data, val_data, test_data]):
            for pt in data:
                if pt['labels'] is None:
                    continue
                label = {'boxes': [], 'labels': [], 'name': pt['name']}

                for lab in pt['labels']:
                    if not lab['category'] in self.CATEGORIES:
                        continue
                    label['boxes'].append(
                        torch.tensor([float(lab['box2d'][c]) for c in ['x1', 'y1', 'x2', 'y2']])
                    )
                    label['labels'].append(self.CATEGORIES[lab['category']])

                if not label['boxes']:
                    continue
                self._image_array.append(pt['name'])
                self._split_array.append(i)
                label['boxes'] = torch.stack(label['boxes'])
                # The above boxes are (x_min,y_min,x_max,y_max)
                # Convert labels into (center_x, center_y, w, h) normalized, which is what DETR expects
                # TODO: If it's not standard, we can probably put this in a transform somewhere
#                 boxes = label['boxes']
#                 center_x = (boxes[:, 0] + boxes[:, 2]) / 2 / self._original_resolution[0]
#                 center_y = (boxes[:, 1] + boxes[:, 3]) / 2 / self._original_resolution[1]
#                 width = (boxes[:, 2] - boxes[:, 0]) / self._original_resolution[0]
#                 height = (boxes[:, 3] - boxes[:, 1]) / self._original_resolution[1]
#                 label['boxes'] = torch.stack((center_x, center_y, width, height), dim=1)
                label['labels'] = torch.tensor(label['labels']).long()
                self._y_array.append(label)
                self._metadata_array.append(self.GROUPS[pt['attributes']['timeofday']])
        self._split_array = np.array(self._split_array)
        self._metadata_array = torch.tensor(self._metadata_array,
                                            dtype=torch.long).unsqueeze(1)
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['timeofday'],
        )
        self._metric = MulticlassDetectionAccuracy(
            id_to_cat={v: k for k, v in self.CATEGORIES.items()}
        )
        super(BDD100KDetDataset, self).__init__(version=version,
                                                root_dir=root_dir,
                                                download=False,
                                                split_scheme='official')

    def eval(self, y_pred, y_true, metadata, aggregate=True):
        """
        This method operates somewhat differently from `WILDSDataset.standard_group_eval`.
        Specifically, `self._eval_grouper` does not include 'y' since, for detection,
        each image has multiple labels, so the grouping interface is not the same.
        Instead, `self._eval_grouper` only groups by 'timeofday', which appropriately
        assigns a single metadata value to each image, as the grouper expects.
        To also group the results by labels, we will define `self._metric.compute_group_wise`
        such that it accepts predictions, targets, and groups, and it returns results
        separately for each label and group.
        """
        metric, grouper = self._metric, self._eval_grouper
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups, self._n_classes)
        for group_idx in range(grouper.n_groups):
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            for cls in range(self._n_classes):
                group_str = grouper.group_field_str(group_idx) + f'_y:{cls}'
                group_metric = group_results[metric.group_metric_field(group_idx, cls)]
                results[f'{metric.name}_{group_str}'] = group_metric
                results_str += (
                    f'  {grouper.group_str(group_idx)}, y = {cls}  '
                    f"{metric.name} = {group_results[metric.group_metric_field(group_idx, cls)]:5.3f}\n")
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f'count_{grouper.group_field_str(group_idx)}'] = group_counts
            results_str += (
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
            )
        results[f'{metric.worst_group_metric_field}'] = group_results[f'{metric.worst_group_metric_field}']
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str

    def _collate(self, batch):
        #TODO: make this a util?
        batch = list(zip(*batch))
        batch[0] = torch.stack(batch[0])
        batch[2] = torch.stack(batch[2])
        return tuple(batch)


class BDD100KClsDataset(BDD100KDataset):
    """
    The BDD100K-wilds classification driving dataset.
    This is a modified version of the original BDD100K dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to reproduce observations discussed in the WILDS paper.

    Supported `split_scheme`:
        'official', 'timeofday' (equivalent to 'official'), or 'location'

    Metadata:
        If `split_scheme` is 'official' or 'timeofday', each data point is
        annotated with a time of day from `BDD100KDataset.TIMEOFDAY_SPLITS`.
        If `split_scheme` is 'location' each data point is annotated with a
        location from `BDD100KDataset.LOCATION_SPLITS`.

    Output (y):
        `y` is a 9-dimensional binary vector that is `1` at index `i` if
        `BDD100KClsDataset.CATEGORIES[i]` is present in the image and `0` otherwise.
    """
    CATEGORIES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'rider',
                  'traffic light', 'traffic sign', 'truck']
    TIMEOFDAY_SPLITS = ['daytime', 'night', 'dawn/dusk', 'undefined']
    LOCATION_SPLITS = ['New York', 'California']

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (1280, 720)
        self.root = Path(self.data_dir)

        if split_scheme in ('official', 'timeofday'):
            self._to_load = 'timeofday'
        elif split_scheme == 'location':
            self._to_load = 'location'
        else:
            raise ValueError("For BDD100K-wilds classification, split scheme should be "
                             "'official', 'timeofday', or 'location'.")
        self._metadata_fields = [self._to_load]
        self._split_scheme = split_scheme
        train_data_df = pd.read_csv(self.root / f'{self._to_load}_train.csv')
        val_data_df = pd.read_csv(self.root / f'{self._to_load}_val.csv')
        test_data_df = pd.read_csv(self.root / f'{self._to_load}_test.csv')
        self._image_array = []
        self._split_array, self._y_array, self._metadata_array = [], [], []

        for i, df in enumerate([train_data_df, val_data_df, test_data_df]):
            self._image_array.extend(list(df['image'].values))
            labels = [list(df[cat].values) for cat in self.CATEGORIES]
            labels = list(zip(*labels))
            self._split_array.extend([i] * len(labels))
            self._y_array.extend(labels)
            self._metadata_array.extend(list(df['group'].values))
        self._y_size = len(self.CATEGORIES)
        self._split_array = np.array(self._split_array)
        self._y_array = torch.tensor(self._y_array, dtype=torch.float)
        self._metadata_array = torch.tensor(self._metadata_array,
                                            dtype=torch.long).unsqueeze(1)
        split_names = (self.TIMEOFDAY_SPLITS if split_to_load == 'timeofday'
                       else self.LOCATION_SPLITS)
        self._metadata_map = {split_to_load: split_names}
        self._metric = MultiTaskAccuracy()
        super(BDD100ClsDataset, self).__init__(version=version,
                                               root_dir=root_dir,
                                               download=download,
                                               split_scheme=split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels 
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        results = self._metric.compute(y_pred, y_true)
        results_str = (f'{self._metric.name}: '
                       f'{results[self._metric.agg_metric_field]:.3f}\n')
        return results, results_str
