# Code adapted from https://github.com/liucong3/camelyon17
# and https://github.com/cv-lee/Camelyon17

import openslide
import cv2
import numpy as np
import pandas as pd
import os
import csv
import argparse
from tqdm import tqdm

from xml.etree.ElementTree import parse
from PIL import Image

PATCH_LEVEL = 2
MASK_LEVEL = 4
CENTER_SIZE = 32

def _read_xml(xml_path, mask_level):
    """
    Read an XML file with annotations and return coordinates of tumor and normal areas
    """

    xml = parse(xml_path).getroot()

    tumor_coord_list = []
    normal_coord_list = []
    for annotation in xml.iter('Annotation'):
        annotation_type = annotation.get('PartOfGroup')
        assert annotation_type in ['metastases', 'normal', 'None']
        if annotation_type == 'metastases':
            coord_list = tumor_coord_list
        elif annotation_type == 'normal':
            coord_list = normal_coord_list
        elif annotation_type == 'None':
            continue

        for region_idx, region in enumerate(annotation.iter('Coordinates')):
            assert region_idx == 0
            coords = []
            for coord in region:
                coords.append([round(float(coord.get('X'))/(2**mask_level)),
                               round(float(coord.get('Y'))/(2**mask_level))])
            coord_list.append(coords)

    return tumor_coord_list, normal_coord_list

def _make_masks(slide_path, xml_path, mask_level, make_map, **args):
    '''
    Return a slide with annotated tumor, normal, and tissue masks using an Otsu threshold
    '''
    print('_make_masks(%s)' % slide_path)

    #slide loading
    slide = openslide.OpenSlide(slide_path)
    # xml loading
    tumor_coord_list, normal_coord_list = _read_xml(xml_path, mask_level)

    if make_map:
        slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))
        # draw boundary of tumor in map
        for coords in tumor_coord_list:
            cv2.drawContours(slide_map, np.array([coords]), -1, 255, 1)
        for coords in normal_coord_list:
            cv2.drawContours(slide_map, np.array([coords]), -1, 127, 1)
    else:
        slide_map = None

    # draw tumor mask
    # first fill up tumors, then draw normal boundaries and fill those up with 0
    tumor_mask = np.zeros(slide.level_dimensions[mask_level][::-1])
    for coords in tumor_coord_list:
        cv2.drawContours(tumor_mask, np.array([coords]), -1, 255, -1)
    for coords in normal_coord_list:
        cv2.drawContours(tumor_mask, np.array([coords]), -1, 0, -1)

    # draw tissue mask
    slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
    slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
    slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
    slide_lv = slide_lv[:, :, 1]
    _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # check normal mask / draw normal mask
    normal_mask = np.array(tissue_mask).copy()
    normal_mask[tumor_mask > 127] = 0

    return slide, slide_map, tumor_mask, tissue_mask, normal_mask


def _write_masks(mask_folder_path, slide_map, tumor_mask, tissue_mask, normal_mask, **args):
    """
    Write masks out to disk; used for sanity checking and visualization.
    """
    print('_write_masks')
    os.makedirs(mask_folder_path, exist_ok=True)
    map_path = os.path.join(mask_folder_path, 'map.png')
    cv2.imwrite(map_path, slide_map)
    tumor_mask_path = os.path.join(mask_folder_path, 'tumor_mask.png')
    cv2.imwrite(tumor_mask_path, tumor_mask) # CHANGED
    tissue_mask_path = os.path.join(mask_folder_path, 'tissue_mask.png')
    cv2.imwrite(tissue_mask_path, np.array(tissue_mask))
    normal_mask_path = os.path.join(mask_folder_path, 'normal_mask.png')
    cv2.imwrite(normal_mask_path, normal_mask)


def _record_patches(center_size,
                    slide, slide_map, patch_level,
                    mask_level, tumor_mask, tissue_mask, normal_mask,
                    tumor_threshold,
                    normal_threshold,                    
                    **args):
    """
    Extract all tumor and non-tumor patches from a slide, using the given masks.
    """

    # Patch size is 3*center_size by 3*center_size
    # It is in terms of pixels of the final output
    # So it's measured with respect to patch_level
    patch_size = center_size * 3

    # Extract normal, tumor patches using normal, tumor mask
    width, height = np.array(slide.level_dimensions[patch_level]) // center_size
    total = width * height
    all_cnt = 0
    t_cnt = 0
    n_cnt = 0

    print('_record_patches(w=%d,h=%d)' % (width,height))
    margin = 5 #3
    mask_max = 255
    assert mask_level >= patch_level
    width_mask_step = center_size * slide.level_dimensions[mask_level][0] / slide.level_dimensions[patch_level][0]
    height_mask_step = center_size * slide.level_dimensions[mask_level][1] / slide.level_dimensions[patch_level][1]

    patch_list = []

    # These mark the coordinates of the central region of the patch
    for i in range(margin, width-margin):
        for j in range(margin, height-margin):

            mask_i_start = round(width_mask_step * i)
            mask_i_end = round(width_mask_step * (i+1))
            mask_j_start = round(height_mask_step * j)
            mask_j_end = round(height_mask_step * (j+1))

            # Compute masks only over central region
            tumor_mask_avg = tumor_mask[
                mask_j_start : mask_j_end,
                mask_i_start : mask_i_end].mean()
            normal_mask_avg = normal_mask[
                mask_j_start : mask_j_end,
                mask_i_start : mask_i_end].mean()

            tumor_area_ratio = tumor_mask_avg / mask_max
            normal_area_ratio = normal_mask_avg / mask_max

            # Extract patch coordinates
            # Coords correspond just to the center, not the entire patch
            if (tumor_area_ratio > tumor_threshold):
                patch_list.append((center_size*i, center_size*j, 1))
                cv2.rectangle(
                    slide_map,
                    (mask_i_start, mask_j_start),
                    (mask_i_end, mask_j_end),
                    (0,0,255),
                    1)

            elif (normal_area_ratio > normal_threshold):
                patch_list.append((center_size*i, center_size*j, 0))
                cv2.rectangle(
                    slide_map,
                    (mask_i_start, mask_j_start),
                    (mask_i_end, mask_j_end),
                    (255,255,0),
                    1)

    df = pd.DataFrame(patch_list,
        columns=[
            'x_coord',
            'y_coord',
            'tumor'
        ])
    return df


def generate_file(patient, node, xml_path, slide_path, folder_path):
    args = {
        'slide_path' : slide_path,
        'xml_path': xml_path,
        'patch_level' : PATCH_LEVEL,
        'mask_level' : MASK_LEVEL,
        'center_size' : CENTER_SIZE,
        'tumor_threshold' : 0,
        'normal_threshold' : 0.2,
        'mask_folder_path' : folder_path,
        'make_map' : True
    }
    args['slide'], args['slide_map'], args['tumor_mask'], args['tissue_mask'], args['normal_mask'] = _make_masks(**args)
    df = _record_patches(**args)
    df['patient'] = patient
    df['node'] = node
    _write_masks(**args)

    return df


def generate_files(slide_root, output_root):
    aggregate_df = pd.DataFrame(
        columns=[
            'patient',
            'node',
            'x_coord',
            'y_coord',
            'tumor'
        ])

    for root, dirs, files in os.walk(os.path.join(slide_root, 'lesion_annotations')):
        for file in files:
            if file.endswith('.xml') and not file.startswith('._'):
                prefix = file.split('.xml')[0]
                try:
                    assert len(prefix.split('_')) == 4
                    df = generate_file(
                        patient=prefix.split('_')[1],
                        node=prefix.split('_')[3],
                        xml_path=os.path.join(root, file),
                        slide_path=os.path.join(slide_root, 'tif', f'{prefix}.tif'),
                        folder_path=os.path.join(output_root, 'masks', prefix))
                    aggregate_df = pd.concat([aggregate_df, df])

                except openslide.OpenSlideError as err:
                    print(err)
                    continue

    aggregate_df = aggregate_df.reset_index(drop=True)
    aggregate_df.to_csv(os.path.join(output_root, 'all_patch_coords.csv'))
    return aggregate_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--slide_root', required=True)
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()

    generate_files(
        slide_root=args.slide_root,
        output_root=args.output_root)
