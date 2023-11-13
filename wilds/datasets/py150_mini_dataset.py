from wilds.datasets.py150_dataset import Py150Dataset
import pandas as pd

class Py150MiniDataset(Py150Dataset):
    def __init__(self, mini_size=1000, *args, **kwargs):
        self.mini_size = mini_size
        super().__init__(*args, **kwargs)

    def _load_all_data(self):
        # Load the full dataset
        full_df = super()._load_all_data()

        # Shuffle the DataFrame
        shuffled_df = full_df.sample(frac=1).reset_index(drop=True)

        # Select the first `mini_size` rows
        mini_df = shuffled_df.head(self.mini_size)

        return mini_df
