## Unlabeled Camelyon17-WILDS patch processing

#### Requirements

- openslide-python>=1.1.2
- opencv-python>=4.4.0

openslide-python relies on first installing OpenSlide; 
see [installation instructions](https://github.com/openslide/openslide-python).

#### Instructions

1. Download the [CAMELYON17 training data](https://drive.google.com/drive/folders/0BzsdkU4jWx9BSEI2X1VOLUpYZ3c?resourcekey=0-41XIPJNyEAo598wHxVAP9w) 
   into `SLIDE_ROOT`.

2. Run `python generate_all_patch_coords.py --slide_root SLIDE_ROOT --output_root OUTPUT_ROOT` to generate a .csv of all 
   potential patches as well as the tissue masks for each WSI. `OUTPUT_ROOT` is wherever you would like the 
   patches to eventually be written.

3. Then run `python generate_final_metadata.py --slide_root SLIDE_ROOT --output_root OUTPUT_ROOT` 
   to generate the metadata.csv file for unlabeled Camelyon.
   
4. Finally, run `python extract_final_patches_to_disk.py --slide_root SLIDE_ROOT --output_root OUTPUT_ROOT` to 
   extract the chosen patches from the WSIs and write them to disk.