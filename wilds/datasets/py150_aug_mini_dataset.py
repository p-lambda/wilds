from wilds.datasets.py150_aug_dataset import Py150AugDataset
import pandas as pd

class Py150AugMiniDataset(Py150AugDataset):
    def __init__(self, mini_size=1000, *args, **kwargs):
        self.mini_size = mini_size
        super().__init__(*args, **kwargs)

    def _load_all_data(self):
        # Load the full dataset
        full_df = super()._load_all_data()

        print(50 * "=")
        print("aug-mini:", full_df)
        print(50 * "=")

        # Shuffle the DataFrame
        shuffled_df = full_df.sample(frac=1).reset_index(drop=True)

        # Select the first `mini_size` rows
        mini_df = shuffled_df.head(self.mini_size)

        return mini_df
