## Camelyon17-wilds patch processing

#### Requirements
- openslide-python>=1.1.2
- opencv-python>=4.4.0

openslide-python relies on first installing OpenSlide; 
see [installation instructions](https://github.com/openslide/openslide-python).

#### Instructions

1. Download the CAMELYON17 data from https://camelyon17.grand-challenge.org/Data/ into `SLIDE_ROOT`. 
   The dataset is huge, so you might want to only download the 100 WSIs with lesion annotations, which by themselves are already 600G. 
   You can find out which WSIs have annotations by looking at the `lesion_annotations` folder. The patch extraction code expects 
   `SLIDE_ROOT` to contain the `lesion_annotations` and `tif` folders.
   
    expected: camelyon/tif/patient_021_node_3.tif?

2. Run `python generate_all_patch_coords.py --slide_root SLIDE_ROOT --output_root OUTPUT_ROOT` to generate a .csv of all 
   potential patches as well as the tissue/tumor/normal masks for each WSI. `OUTPUT_ROOT` is wherever you would like the 
   patches to eventually be written. `python generate_all_patch_coords.py --slide_root camelyon --output_root output`

3. Then run `python generate_final_metadata.py --output_root OUTPUT_ROOT` to select a class-balanced set of patches 
   and assign splits.

4. Finally, run `python extract_final_patches_to_disk.py --slide_root SLIDE_ROOT --output_root OUTPUT_ROOT` to 
   extract the chosen patches from the WSIs and write them to disk.


Make sure to exclude the slides that are in train.


Camelyon
Dataset: https://drive.google.com/drive/folders/0BzsdkU4jWx9BSEI2X1VOLUpYZ3c?resourcekey=0-41XIPJNyEAo598wHxVAP9w
Splits
From WILDS:
test_center = 2
val_center = 1
TIFF to png
https://camelyon17.grand-challenge.org/Data/

Input: 96x96 histopathological image
Label: y that is a binary indicator of whether the central 32x32 region contains the tumor or not
Dataset: 450,000 patches extracted from 50 whole-slide images (WSIs). 10 WSIs from each hospital.
Train: 1st, 2nd and 3rd hospitals
Validation: 4th hospital
Test: 5th hospital

TODO: modify generate_final_metadata and extract_final_patches_to_disk scripts.



Questions for Pang Wei:
- Should we not preprocess data in the lesion_annotations folder?
    metadata.csv instead
- For generate_all_patch_coords: expected: camelyon/tif/patient_021_node_3.tif?
- class balancing done in generate_final_metadata: subsample the 3000 patches from each slide:
- Keep extract_final_patches_to_disk.py intact? yes
- 200 slides from each hospital. 1 million patches from each center.


XML path contains the labeled data and the location of the tumors. We just want to use otsu threshold.
call generate_file on each file, but skip distinguishing tumor or not tumor.
300,000 patches 10x = 3 million patches 
