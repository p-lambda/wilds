import argparse
import os
import pdb

"""
Validate the content of the unlabeled Camelyon17 dataset after 
preprocessing and uploading to CodaLab.

Statistics:
Total # of patches: 2,999,307
Center 0: 599,187 patches
Center 1: 600,030 patches
Center 2: 600,030 patches
Center 3: 600,030 patches
Center 4: 600,030 patches

Usage:

    python dataset_preprocessing/camelyon17/unlabeled/validate.py <root_dir>
"""

_EXPECTED_SLIDES_COUNT = 450


def validate_unlabeled_dataset(root_dir: str):
    def get_patients_center(patient_id: str):
        patient_no = int(patient_id)
        if 0 <= patient_no < 20:
            return 0
        elif 20 <= patient_no < 40:
            return 1
        elif 40 <= patient_no < 60:
            return 2
        elif 60 <= patient_no < 80:
            return 3
        elif 80 <= patient_no < 100:
            return 4
        else:
            raise ValueError(f"Can't get center for patient {patient_id}.")

    dataset_dir = os.path.join(root_dir, "camelyon17_unlabeled_v1.0")
    content = os.listdir(dataset_dir)
    assert "patches" in content
    assert "RELEASE_v1.0.txt" in content
    assert "metadata.csv" in content

    slides_dir = os.path.join(dataset_dir, "patches")
    slides = os.listdir(slides_dir)

    slide_count = 0
    patch_counts = [0 for _ in range(5)]
    for slide in slides:
        patches_dir = os.path.join(slides_dir, slide)
        if not os.path.isdir(patches_dir):
            continue
        slide_count += 1

        slide_split = slide.split("_")
        assert len(slide_split) == 4
        patient_id = slide_split[1]
        center = get_patients_center(patient_id)
        for patch in os.listdir(patches_dir):
            if patch.endswith(".png"):
                patch_counts[center] += 1

    assert (
        slide_count == _EXPECTED_SLIDES_COUNT
    ), f"Got incorrect number of slides. Expected: {_EXPECTED_SLIDES_COUNT}, Actual: {len(slides)}"
    print(f"Patch counts: {patch_counts}")
    assert patch_counts == [599187, 600030, 600030, 600030, 600030]
    assert sum(patch_counts) == 2999307
    print("\nVerified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Path to the datasets directory.")
    args = parser.parse_args()
    validate_unlabeled_dataset(args.root_dir)
