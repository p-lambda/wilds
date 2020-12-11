'''
Adapted from github.com/sustainlab-group/africa_poverty/data_analysis/dhs.ipynb
'''
import tensorflow as tf
import numpy as np
import batcher
import dataset_constants
from tqdm import tqdm

FOLDS = ['A', 'B', 'C', 'D', 'E']
SPLITS = ['train', 'val', 'test']
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
DATASET = '2009-17'

COUNTRIES = np.asarray(dataset_constants.DHS_COUNTRIES)


def get_images(tfrecord_paths, label_name='wealthpooled', return_meta=False):
    '''
    Args
    - tfrecord_paths: list of str, length N <= 32, paths of TFRecord files

    Returns: np.array, shape [N, 224, 224, 8], type float32
    '''
    init_iter, batch_op = batcher.Batcher(
        tfrecord_files=tfrecord_paths,
        dataset=DATASET,
        batch_size=32,
        ls_bands='ms',
        nl_band='merge',
        label_name=label_name,
        shuffle=False,
        augment=False,
        negatives='zero',
        normalize=True).get_batch()
    with tf.Session() as sess:
        sess.run(init_iter)
        if return_meta:
            ret = sess.run(batch_op)
        else:
            ret = sess.run(batch_op['images'])
    return ret


if __name__ == '__main__':
    tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(dataset=DATASET, split='all'))

    num_batches = len(tfrecord_paths) // 32
    if len(tfrecord_paths) % 32 != 0:
        num_batches += 1

    imgs = []

    for i in tqdm(range(num_batches)):
        imgs.append(get_images(tfrecord_paths[i*32: (i+1)*32]))

    imgs = np.concatenate(imgs, axis=0)
    np.save('/scr/landsat_poverty_imgs.npy', imgs)


    ######### process unlabeled data

    tfrecord_paths = []
    root = Path('/atlas/u/chrisyeh/poverty_data/lxv3_transfer')
    for country_year in root.iterdir():
        if not country_year.is_dir():
            continue
        for tfrecord_file in country_year.iterdir():
            tfrecord_paths.append(str(tfrecord_file))

    batch_size = 32
    num_batches = len(tfrecord_paths) // batch_size
    if len(tfrecord_paths) % batch_size != 0:
        num_batches += 1

    metadata = []
    imgs = []

    counter = 0
    for i in tqdm(range(num_batches)):
        batch_paths = tfrecord_paths[i*batch_size: (i+1)*batch_size]
        img_batch = get_images(batch_paths, label_name=None, return_meta=True)
        nl_means = img_batch['images'][:, :, :, -1].mean((1,2))
        nl_centers = img_batch['images'][:, 112, 112, -1]

        for path, loc, year, nl_mean, nl_center in zip(batch_paths, img_batch['locs'], img_batch['years'], nl_means, nl_centers):
            country = "_".join(str(Path(path).parent.stem).split('_')[:-1])

            metadata.append({'country': country, 'lat': loc[0], 'lon': loc[1], 'year': year, 'nl_mean': float(nl_mean), 'nl_center': float(nl_center)})

        imgs.append(img_batch['images'])

        if len(imgs) > (10000 // 32):
            imgs = np.concatenate(imgs, axis=0)
            np.save(f'/u/scr/nlp/dro/poverty/unlabeled_landsat_poverty_imgs_{counter}.npy', imgs)
            counter += 1
            imgs = []
    if len(imgs) > 0:
        imgs = np.concatenate(imgs, axis=0)
        np.save(f'/u/scr/nlp/dro/poverty/unlabeled_landsat_poverty_imgs_{counter}.npy', imgs)

    df = pd.DataFrame(metadata)
    df.to_csv('/u/scr/nlp/dro/poverty/unlabeled_metadata.csv', index=False)
