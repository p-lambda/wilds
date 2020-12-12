## Camelyon17-wilds patch processing

#### Requirements
- openslide-python>=1.1.2
- opencv-python>=4.4.0

openslide-python relies on first installing OpenSlide; see [installation instructions](https://github.com/openslide/openslide-python).

#### Instructions

1. Download the CAMELYON17 data from https://camelyon17.grand-challenge.org/Data/ into `SLIDE_ROOT`. The dataset is huge, so you might want to only download the 100 WSIs with lesion annotations, which by themselves are already 600G. You can find out which WSIs have annotations by looking at the `lesion_annotations` folder. The patch extraction code expects `SLIDE_ROOT` to contain the `lesion_annotations` and `tif` folders.

2. Run `python generate_all_patch_coords.py --slide_root SLIDE_ROOT --output_root OUTPUT_ROOT` to generate a .csv of all potential patches as well as the tissue/tumor/normal masks for each WSI. `OUTPUT_ROOT` is wherever you would like the patches to eventually be written.

3. Then run `python generate_final_metadata.py --output_root OUTPUT_ROOT` to select a class-balanced set of patches and assign splits.

4. Finally, run `python extract_final_patches_to_disk.py --slide_root SLIDE_ROOT --output_root OUTPUT_ROOT` to extract the chosen patches from the WSIs and write them to disk.
