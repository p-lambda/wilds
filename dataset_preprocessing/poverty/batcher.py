########
# ADAPTED from github.com/sustainlab-group/africa_poverty
########
from dataset_constants import SIZES, SURVEY_NAMES, MEANS_DICT, STD_DEVS_DICT

from glob import glob
import os

import tensorflow as tf


ROOT_DIR = '/atlas/u/chrisyeh/africa_poverty/'
DHS_TFRECORDS_PATH_ROOT = os.path.join(ROOT_DIR, 'data/dhs_tfrecords')
LSMS_TFRECORDS_PATH_ROOT = os.path.join(ROOT_DIR, 'data/lsms_tfrecords')


def get_tfrecord_paths(dataset, split='all'):
    '''
    Args
    - dataset: str, a key in SURVEY_NAMES
    - split: str, one of ['train', 'val', 'test', 'all']

    Returns:
    - tfrecord_paths: list of str, paths to TFRecord files, sorted
    '''
    expected_size = SIZES[dataset][split]
    if split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [split]

    survey_names = SURVEY_NAMES[dataset]
    tfrecord_paths = []
    for split in splits:
        for country_year in survey_names[split]:
            glob_path = os.path.join(DHS_TFRECORDS_PATH_ROOT, country_year + '*', '*.tfrecord.gz')
            tfrecord_paths.extend(glob(glob_path))
    tfrecord_paths = sorted(tfrecord_paths)
    assert len(tfrecord_paths) == expected_size
    return tfrecord_paths


def get_lsms_tfrecord_paths(cys):
    '''
    Args
    - cys: list of 'country_year' str, order matters!

    Returns:
    - tfrecord_paths: list of str, paths to TFRecord files, order of country_years given by cys
    '''
    expected_size = sum([SIZES['LSMS'][cy] for cy in cys])
    tfrecord_paths = []
    for cy in cys:
        glob_path = os.path.join(LSMS_TFRECORDS_PATH_ROOT, cy, '*.tfrecord.gz')
        tfrecord_paths.extend(sorted(glob(glob_path)))
    assert len(tfrecord_paths) == expected_size
    return tfrecord_paths


class Batcher():
    def __init__(self, tfrecord_files, dataset, batch_size, label_name,
                 num_threads=1, epochs=1, ls_bands='rgb', nl_band=None, nl_label=None,
                 shuffle=True, augment=True, negatives='zero', normalize=True, cache=False):
        '''
        Args
        - tfrecord_files: str, list of str, or a tf.Tensor (e.g. tf.placeholder) of str
            - path(s) to TFRecord files containing satellite images
        - dataset: str, one of the keys of MEANS_DICT
        - batch_size: int
        - label_name: str, name of feature within TFRecords of labels, or None
        - epochs: int, number of epochs to repeat the dataset
        - ls_bands: one of [None, 'rgb', 'ms']
            - None: no Landsat bands
            - 'rgb': only the RGB bands
            - 'ms': all 7 Landsat bands
        - nl_band: one of [None, 'merge', 'split']
            - None: no nightlights band
            - 'merge': single nightlights band
            - 'split': separate bands for DMSP and VIIRS (if one is absent, then band is all 0)
        - nl_label: one of [None, 'center', 'mean']
            - None: do not include nightlights as a label
            - 'center': nightlight value of center pixel
            - 'mean': mean nightlights value
        - shuffle: bool, whether to shuffle data, should be False when not training
        - augment: bool, whether to use data augmentation, should be False when not training
        - negatives: one of [None, 'zero'], what to do with unexpected negative values
            - None: do nothing (keep the negative values)
            - 'zero': clip the negative values to 0
        - normalize: bool, whether to subtract mean and divide by std_dev
        - cache: bool, whether to cache this dataset in memory
        '''
        self.tfrecord_files = tfrecord_files
        self.batch_size = batch_size
        self.label_name = label_name
        self.num_threads = num_threads
        self.epochs = epochs
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        self.cache = cache

        if ls_bands not in [None, 'rgb', 'ms']:
            raise ValueError(f'Error: got {ls_bands} for "ls_bands"')
        self.ls_bands = ls_bands

        if dataset not in MEANS_DICT:
            raise ValueError(f'Error: got {dataset} for "dataset"')
        self.dataset = dataset

        if negatives not in [None, 'zero']:
            raise ValueError(f'Error: got {negatives} for "negatives"')
        self.negatives = negatives

        if nl_band not in [None, 'merge', 'split']:
            raise ValueError(f'Error: got {nl_band} for "nl_band"')
        self.nl_band = nl_band

        if nl_label not in [None, 'center', 'mean']:
            raise ValueError(f'Error: got {nl_label} for "nl_label"')
        self.nl_label = nl_label

    def get_batch(self):
        '''Gets the tf.Tensors that represent a batch of data.

        Returns
        - iter_init: tf.Operation that should be run before each epoch
        - batch: dict, str -> tf.Tensor
            - 'images': tf.Tensor, shape [batch_size, H, W, C], type float32
                - C depends on the ls_bands and nl_band settings
            - 'locs': tf.Tensor, shape [batch_size, 2], type float32, each row is [lat, lon]
            - 'labels': tf.Tensor, shape [batch_size] or [batch_size, label_dim], type float32
                - shape [batch_size, 2] if self.label_name and self.nl_label are not None
            - 'years': tf.Tensor, shape [batch_size], type int32

        IMPLEMENTATION NOTE: The order of tf.data.Dataset.batch() and .repeat() matters!
            Suppose the size of the dataset is not evenly divisible by self.batch_size.
            If batch then repeat, ie. `ds.batch(batch_size).repeat(num_epochs)`:
                the last batch of every epoch will be smaller than batch_size
            If repeat then batch, ie. `ds.repeat(num_epochs).batch(batch_size)`:
                the boundaries between epochs are blurred, ie. the dataset "wraps around"
        '''
        if self.shuffle:
            # shuffle the order of the input files, then interleave their individual records
            dataset = tf.data.Dataset.from_tensor_slices(self.tfrecord_files)
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.apply(tf.contrib.data.parallel_interleave(
                lambda file_path: tf.data.TFRecordDataset(file_path, compression_type='GZIP'),
                cycle_length=self.num_threads,
                block_length=1
            ))
        else:
            # convert to individual records
            dataset = tf.data.TFRecordDataset(
                filenames=self.tfrecord_files,
                compression_type='GZIP',
                buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
                num_parallel_reads=self.num_threads)

        # filter out unwanted TFRecords
        if getattr(self, 'filter_fn', None) is not None:
            dataset = dataset.filter(self.filter_fn)

        # prefetch 2 batches at a time to smooth out the time taken to
        # load input files as we go through shuffling and processing
        dataset = dataset.prefetch(buffer_size=2 * self.batch_size)
        dataset = dataset.map(self.process_tfrecords, num_parallel_calls=self.num_threads)
        if self.nl_band == 'split':
            dataset = dataset.map(self.split_nl_band)

        if self.cache:
            dataset = dataset.cache()
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        if self.augment:
            dataset = dataset.map(self.augment_example)

        # batch then repeat => batches respect epoch boundaries
        # - i.e. last batch of each epoch might be smaller than batch_size
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.epochs)

        # prefetch 2 batches at a time
        dataset = dataset.prefetch(2)

        iterator = dataset.make_initializable_iterator()
        batch = iterator.get_next()
        iter_init = iterator.initializer
        return iter_init, batch

    def process_tfrecords(self, example_proto):
        '''
        Args
        - example_proto: a tf.train.Example protobuf

        Returns: dict {'images': img, 'labels': label, 'locs': loc, 'years': year}
        - img: tf.Tensor, shape [224, 224, C], type float32
          - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
        - label: tf.Tensor, scalar or shape [2], type float32
          - not returned if both self.label_name and self.nl_label are None
          - [label, nl_label] (shape [2]) if self.label_name and self.nl_label are both not None
          - otherwise, is a scalar tf.Tensor containing the single label
        - loc: tf.Tensor, shape [2], type float32, order is [lat, lon]
        - year: tf.Tensor, scalar, type int32
          - default value of -1 if 'year' is not a key in the protobuf
        '''
        bands = []
        if self.ls_bands == 'rgb':
            bands = ['BLUE', 'GREEN', 'RED']  # BGR order
        elif self.ls_bands == 'ms':
            bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR']
        if self.nl_band is not None:
            bands += ['NIGHTLIGHTS']

        scalar_float_keys = ['lat', 'lon', 'year']
        if self.label_name is not None:
            scalar_float_keys.append(self.label_name)

        keys_to_features = {}
        for band in bands:
            keys_to_features[band] = tf.FixedLenFeature(shape=[255**2], dtype=tf.float32)
        for key in scalar_float_keys:
            keys_to_features[key] = tf.FixedLenFeature(shape=[], dtype=tf.float32)

        ex = tf.parse_single_example(example_proto, features=keys_to_features)
        loc = tf.stack([ex['lat'], ex['lon']])
        year = tf.cast(ex.get('year', -1), tf.int32)

        img = float('nan')
        if len(bands) > 0:
            means = MEANS_DICT[self.dataset]
            std_devs = STD_DEVS_DICT[self.dataset]

            # for each band, subtract mean and divide by std dev
            # then reshape to (255, 255) and crop to (224, 224)
            for band in bands:
                ex[band].set_shape([255 * 255])
                ex[band] = tf.reshape(ex[band], [255, 255])[15:-16, 15:-16]
                if self.negatives == 'zero':
                    ex[band] = tf.nn.relu(ex[band])
                if self.normalize:
                    if band == 'NIGHTLIGHTS':
                        ex[band] = tf.cond(
                            year < 2012,  # true = DMSP
                            true_fn=lambda: (ex[band] - means['DMSP']) / std_devs['DMSP'],
                            false_fn=lambda: (ex[band] - means['VIIRS']) / std_devs['VIIRS']
                        )
                    else:
                        ex[band] = (ex[band] - means[band]) / std_devs[band]
            img = tf.stack([ex[band] for band in bands], axis=2)

        result = {'images': img, 'locs': loc, 'years': year}

        if self.nl_label == 'mean':
            nl_label = tf.reduce_mean(ex['NIGHTLIGHTS'])
        elif self.nl_label == 'center':
            nl_label = ex['NIGHTLIGHTS'][112, 112]

        if self.label_name is None:
            if self.nl_label is None:
                label = None
            else:
                label = nl_label
        else:
            label = ex.get(self.label_name, float('nan'))
            if self.nl_label is not None:
                label = tf.stack([label, nl_label])

        if label is not None:
            result['labels'] = label

        return result

    def split_nl_band(self, ex):
        '''Splits the NL band into separate DMSP and VIIRS bands.

        Args
        - ex: dict {'images': img, 'years': year, ...}
            - img: tf.Tensor, shape [H, W, C], type float32, final band is NL
            - year: tf.Tensor, scalar, type int32

        Returns: ex, with img updated to have 2 NL bands
        - img: tf.Tensor, shape [H, W, C], type float32, last two bands are [DMSP, VIIRS]
        '''
        assert self.nl_band == 'split'
        all_0 = tf.zeros(shape=[224, 224, 1], dtype=tf.float32, name='all_0')
        img = ex['images']
        year = ex['years']

        ex['images'] = tf.cond(
            year < 2012,
            # if DMSP, then add an all-0 VIIRS band to the end
            true_fn=lambda: tf.concat([img, all_0], axis=2),
            # if VIIRS, then insert an all-0 DMSP band before the last band
            false_fn=lambda: tf.concat([img[:, :, 0:-1], all_0, img[:, :, -1:]], axis=2)
        )
        return ex

    def augment_example(self, ex):
        '''Performs image augmentation (random flips + levels adjustments).
        Does not perform level adjustments on NL band(s).

        Args
        - ex: dict {'images': img, ...}
            - img: tf.Tensor, shape [H, W, C], type float32
                NL band depends on self.ls_bands and self.nl_band

        Returns: ex, with img replaced with an augmented image
        '''
        assert self.augment
        img = ex['images']

        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = self.augment_levels(img)

        ex['images'] = img
        return ex

    def augment_levels(self, img):
        '''Perform random brightness / contrast on the image.
        Does not perform level adjustments on NL band(s).

        Args
        - img: tf.Tensor, shape [H, W, C], type float32
            - self.nl_band = 'merge' => final band is NL band
            - self.nl_band = 'split' => last 2 bands are NL bands

        Returns: tf.Tensor with data augmentation applied
        '''
        def rand_levels(image):
            # up to 0.5 std dev brightness change
            image = tf.image.random_brightness(image, max_delta=0.5)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
            return image

        # only do random brightness / contrast on non-NL bands
        if self.ls_bands is not None:
            if self.nl_band is None:
                img = rand_levels(img)
            elif self.nl_band == 'merge':
                img_nonl = rand_levels(img[:, :, :-1])
                img = tf.concat([img_nonl, img[:, :, -1:]], axis=2)
            elif self.nl_band == 'split':
                img_nonl = rand_levels(img[:, :, :-2])
                img = tf.concat([img_nonl, img[:, :, -2:]], axis=2)
        return img


class UrbanBatcher(Batcher):
    def filter_fn(self, example_proto):
        '''
        Args
        - example_proto: a tf.train.Example protobuf

        Returns
        - predicate: tf.Tensor, type bool, True to keep, False to filter out
        '''
        keys_to_features = {
            'urban_rural': tf.FixedLenFeature(shape=[], dtype=tf.float32)
        }
        ex = tf.parse_single_example(example_proto, features=keys_to_features)
        do_keep = tf.equal(ex['urban_rural'], 1.0)
        return do_keep


class RuralBatcher(Batcher):
    def filter_fn(self, example_proto):
        '''
        Args
        - example_proto: a tf.train.Example protobuf

        Returns
        - predicate: tf.Tensor, type bool, True to keep, False to filter out
        '''
        keys_to_features = {
            'urban_rural': tf.FixedLenFeature(shape=[], dtype=tf.float32)
        }
        ex = tf.parse_single_example(example_proto, features=keys_to_features)
        do_keep = tf.equal(ex['urban_rural'], 0.0)
        return do_keep


class ResidualBatcher(Batcher):
    def __init__(self, tfrecord_files, preds_ph, dataset, batch_size, label_name,
                 num_threads=1, epochs=1, ls_bands='rgb', nl_band=None,
                 shuffle=True, augment=True, negatives='zero', normalize=True, cache=False):
        '''
        Args
        - preds_ph: tf.placeholder, for vector of predictions corresponding to the TFRecords
        - see Batcher class for other args
        - does not allow for nl_label
        '''
        self.preds_ph = preds_ph

        super(ResidualBatcher, self).__init__(
            tfrecord_files=tfrecord_files,
            dataset=dataset,
            batch_size=batch_size,
            label_name=label_name,
            num_threads=num_threads,
            epochs=epochs,
            ls_bands=ls_bands,
            nl_band=nl_band,
            nl_label=None,
            shuffle=shuffle,
            augment=augment,
            negatives=negatives,
            normalize=normalize,
            cache=cache)

    def get_batch(self):
        '''Gets the tf.Tensors that represent a batch of data.

        Returns
        - iter_init: tf.Operation that should be run before each epoch
        - batch: dict, str -> tf.Tensor
            - 'images': tf.Tensor, shape [batch_size, H, W, C], type float32
                - C depends on the ls_bands and nl_band settings
            - 'locs': tf.Tensor, shape [batch_size, 2], type float32, each row is [lat, lon]
            - 'labels': tf.Tensor, shape [batch_size], type float32, residuals
            - 'years': tf.Tensor, shape [batch_size], type int32

        IMPLEMENTATION NOTE: The order of tf.data.Dataset.batch() and .repeat() matters!
            Suppose the size of the dataset is not evenly divisible by self.batch_size.
            If batch then repeat, ie. `ds.batch(batch_size).repeat(num_epochs)`:
                the last batch of every epoch will be smaller than batch_size
            If repeat then batch, ie. `ds.repeat(num_epochs).batch(batch_size)`:
                the boundaries between epochs are blurred, ie. the dataset "wraps around"
        '''
        # list of TFRecord file paths => tf.train.Example protos
        tfrecords_ds = tf.data.TFRecordDataset(
            filenames=self.tfrecord_files,
            compression_type='GZIP',
            buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
            num_parallel_reads=self.num_threads
        )
        tfrecords_ds = tfrecords_ds.prefetch(buffer_size=2 * self.batch_size)
        # tf.train.Example proto => {
        #   'images': tf.Tensor, shape [H, W, C], type float32
        #   'labels': tf.Tensor, scalar, type float32, label from TFRecord file
        #   'locs': tf.Tensor, shape [2],  type float32
        #   'years': tf.Tensor, scalar, type int32
        # }
        tfrecords_ds = tfrecords_ds.map(self.process_tfrecords, num_parallel_calls=self.num_threads)

        # tf.Tensor, type float32
        preds_ds = tf.data.Dataset.from_tensor_slices(self.preds_ph)

        # merge the datasets => same as tfrecords_ds, except labels now
        #   refers to the residuals
        dataset = tf.data.Dataset.zip((tfrecords_ds, preds_ds))
        dataset = dataset.map(self.merge_residuals, num_parallel_calls=self.num_threads)

        # if augment, order: cache, shuffle, augment, split NL
        # otherwise, order: split NL, cache, shuffle
        if self.augment:
            if self.cache:
                dataset = dataset.cache()
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.map(self.augment_example)
            if self.nl_band == 'split':
                dataset = dataset.map(self.split_nl_band)
        else:
            if self.nl_band == 'split':
                dataset = dataset.map(self.split_nl_band)
            if self.cache:
                dataset = dataset.cache()
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=1000)

        # batch then repeat => batches respect epoch boundaries
        # - i.e. last batch of each epoch might be smaller than batch_size
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.epochs)

        # prefetch 2 batches at a time
        dataset = dataset.prefetch(2)

        iterator = dataset.make_initializable_iterator()
        batch = iterator.get_next()
        iter_init = iterator.initializer
        return iter_init, batch

    def merge_residuals(self, parsed_dict, pred):
        '''
        Args
        - parsed_dict: dict, contains
          - 'labels': tf.Tensor, scalar, type float32, label from TFRecord file
        - pred: tf.Tensor, scalar, type float32

        Returns
        - parsed_dict: dict, same as input, except 'labels' maps to residual
        '''
        # residual = ground truth - prediction
        parsed_dict['labels'] = parsed_dict['labels'] - pred
        return parsed_dict
