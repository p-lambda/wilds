import csv
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import pandas as pd
from PIL import Image

from wilds.common.utils import map_to_id_array
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

DOMAIN_NET_CATEGORIES = [
    "aircraft_carrier",
    "airplane",
    "alarm_clock",
    "ambulance",
    "angel",
    "animal_migration",
    "ant",
    "anvil",
    "apple",
    "arm",
    "asparagus",
    "axe",
    "backpack",
    "banana",
    "bandage",
    "barn",
    "baseball",
    "baseball_bat",
    "basket",
    "basketball",
    "bat",
    "bathtub",
    "beach",
    "bear",
    "beard",
    "bed",
    "bee",
    "belt",
    "bench",
    "bicycle",
    "binoculars",
    "bird",
    "birthday_cake",
    "blackberry",
    "blueberry",
    "book",
    "boomerang",
    "bottlecap",
    "bowtie",
    "bracelet",
    "brain",
    "bread",
    "bridge",
    "broccoli",
    "broom",
    "bucket",
    "bulldozer",
    "bus",
    "bush",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "calendar",
    "camel",
    "camera",
    "camouflage",
    "campfire",
    "candle",
    "cannon",
    "canoe",
    "car",
    "carrot",
    "castle",
    "cat",
    "ceiling_fan",
    "cello",
    "cell_phone",
    "chair",
    "chandelier",
    "church",
    "circle",
    "clarinet",
    "clock",
    "cloud",
    "coffee_cup",
    "compass",
    "computer",
    "cookie",
    "cooler",
    "couch",
    "cow",
    "crab",
    "crayon",
    "crocodile",
    "crown",
    "cruise_ship",
    "cup",
    "diamond",
    "dishwasher",
    "diving_board",
    "dog",
    "dolphin",
    "donut",
    "door",
    "dragon",
    "dresser",
    "drill",
    "drums",
    "duck",
    "dumbbell",
    "ear",
    "elbow",
    "elephant",
    "envelope",
    "eraser",
    "eye",
    "eyeglasses",
    "face",
    "fan",
    "feather",
    "fence",
    "finger",
    "fire_hydrant",
    "fireplace",
    "firetruck",
    "fish",
    "flamingo",
    "flashlight",
    "flip_flops",
    "floor_lamp",
    "flower",
    "flying_saucer",
    "foot",
    "fork",
    "frog",
    "frying_pan",
    "garden",
    "garden_hose",
    "giraffe",
    "goatee",
    "golf_club",
    "grapes",
    "grass",
    "guitar",
    "hamburger",
    "hammer",
    "hand",
    "harp",
    "hat",
    "headphones",
    "hedgehog",
    "helicopter",
    "helmet",
    "hexagon",
    "hockey_puck",
    "hockey_stick",
    "horse",
    "hospital",
    "hot_air_balloon",
    "hot_dog",
    "hot_tub",
    "hourglass",
    "house",
    "house_plant",
    "hurricane",
    "ice_cream",
    "jacket",
    "jail",
    "kangaroo",
    "key",
    "keyboard",
    "knee",
    "knife",
    "ladder",
    "lantern",
    "laptop",
    "leaf",
    "leg",
    "light_bulb",
    "lighter",
    "lighthouse",
    "lightning",
    "line",
    "lion",
    "lipstick",
    "lobster",
    "lollipop",
    "mailbox",
    "map",
    "marker",
    "matches",
    "megaphone",
    "mermaid",
    "microphone",
    "microwave",
    "monkey",
    "moon",
    "mosquito",
    "motorbike",
    "mountain",
    "mouse",
    "moustache",
    "mouth",
    "mug",
    "mushroom",
    "nail",
    "necklace",
    "nose",
    "ocean",
    "octagon",
    "octopus",
    "onion",
    "oven",
    "owl",
    "paintbrush",
    "paint_can",
    "palm_tree",
    "panda",
    "pants",
    "paper_clip",
    "parachute",
    "parrot",
    "passport",
    "peanut",
    "pear",
    "peas",
    "pencil",
    "penguin",
    "piano",
    "pickup_truck",
    "picture_frame",
    "pig",
    "pillow",
    "pineapple",
    "pizza",
    "pliers",
    "police_car",
    "pond",
    "pool",
    "popsicle",
    "postcard",
    "potato",
    "power_outlet",
    "purse",
    "rabbit",
    "raccoon",
    "radio",
    "rain",
    "rainbow",
    "rake",
    "remote_control",
    "rhinoceros",
    "rifle",
    "river",
    "roller_coaster",
    "rollerskates",
    "sailboat",
    "sandwich",
    "saw",
    "saxophone",
    "school_bus",
    "scissors",
    "scorpion",
    "screwdriver",
    "sea_turtle",
    "see_saw",
    "shark",
    "sheep",
    "shoe",
    "shorts",
    "shovel",
    "sink",
    "skateboard",
    "skull",
    "skyscraper",
    "sleeping_bag",
    "smiley_face",
    "snail",
    "snake",
    "snorkel",
    "snowflake",
    "snowman",
    "soccer_ball",
    "sock",
    "speedboat",
    "spider",
    "spoon",
    "spreadsheet",
    "square",
    "squiggle",
    "squirrel",
    "stairs",
    "star",
    "steak",
    "stereo",
    "stethoscope",
    "stitches",
    "stop_sign",
    "stove",
    "strawberry",
    "streetlight",
    "string_bean",
    "submarine",
    "suitcase",
    "sun",
    "swan",
    "sweater",
    "swing_set",
    "sword",
    "syringe",
    "table",
    "teapot",
    "teddy-bear",
    "telephone",
    "television",
    "tennis_racquet",
    "tent",
    "The_Eiffel_Tower",
    "The_Great_Wall_of_China",
    "The_Mona_Lisa",
    "tiger",
    "toaster",
    "toe",
    "toilet",
    "tooth",
    "toothbrush",
    "toothpaste",
    "tornado",
    "tractor",
    "traffic_light",
    "train",
    "tree",
    "triangle",
    "trombone",
    "truck",
    "trumpet",
    "t-shirt",
    "umbrella",
    "underwear",
    "van",
    "vase",
    "violin",
    "washing_machine",
    "watermelon",
    "waterslide",
    "whale",
    "wheel",
    "windmill",
    "wine_bottle",
    "wine_glass",
    "wristwatch",
    "yoga",
    "zebra",
    "zigzag",
]
DOMAIN_NET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
SENTRY_DOMAINS = ["clipart", "painting", "real", "sketch"]


class DomainNetDataset(WILDSDataset):
    """
    DomainNet dataset.
    586,576 images in 345 categories (airplane, ball, cup, etc.) across 6 domains (clipart, infograph, painting,
    quickdraw, real and sketch).

    Supported `split_scheme`:
        'official': use the official split from DomainNet

    Input (x):
        224 x 224 x 3 RGB image.

    Label (y):
        y is one of the 345 categories in DomainNet.

    Metadata:
        None

    Website:
        http://ai.bu.edu/M3SDA

    Original publication:

        @inproceedings{peng2019moment,
          title={Moment matching for multi-source domain adaptation},
          author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
          booktitle={Proceedings of the IEEE International Conference on Computer Vision},
          pages={1406--1415},
          year={2019}
        }

    SENTRY publication:

        @article{prabhu2020sentry
           author = {Prabhu, Viraj and Khare, Shivam and Kartik, Deeksha and Hoffman, Judy},
           title = {SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation},
           year = {2020},
           journal = {arXiv preprint: 2012.11460},
        }

    Fair Use Notice:
        "This dataset contains some copyrighted material whose use has not been specifically authorized by the copyright owners.
        In an effort to advance scientific research, we make this material available for academic research. We believe this
        constitutes a fair use of any such copyrighted material as provided for in section 107 of the US Copyright Law.
        In accordance with Title 17 U.S.C. Section 107, the material on this site is distributed without profit for
        non-commercial research and educational purposes. For more information on fair use please click here. If you wish
        to use copyrighted material on this site or in our dataset for purposes of your own that go beyond non-commercial
        research and academic purposes, you must obtain permission directly from the copyright owner."
    """

    _dataset_name: str = "domainnet"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x0b8ca76eef384b98b879d0c8c4681a32/contents/blob/",
            "compressed_size": 19_255_770_459,
        },
    }

    def __init__(
        self,
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official",
        source_domain: str = "sketch",
        target_domain: str = "real",
        use_sentry: bool = False,
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = self.initialize_data_dir(root_dir, download)

        # The original dataset contains 345 categories. The SENTRY version contains 40 categories.
        if use_sentry:
            assert source_domain in SENTRY_DOMAINS
            assert target_domain in SENTRY_DOMAINS
            print("Using the SENTRY version of DomainNet...")
            metadata_filename = "sentry_metadata.csv"
            self._n_classes = 40
        else:
            metadata_filename = "metadata.csv"
            self._n_classes = 345

        metadata_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, metadata_filename),
            dtype={
                "image_path": str,
                "domain": str,
                "split": str,
                "category": str,
                "y": int,
            },
            keep_default_na=False,
            na_values=[],
            quoting=csv.QUOTE_NONNUMERIC,
        )
        source_metadata_df = metadata_df.loc[metadata_df["domain"] == source_domain]
        target_metadata_df = metadata_df.loc[metadata_df["domain"] == target_domain]
        metadata_df = pd.concat([source_metadata_df, target_metadata_df])

        self._input_image_paths = metadata_df["image_path"].values
        self._y_array = torch.from_numpy(metadata_df["y"].values).type(torch.LongTensor)
        self.initialize_split_dicts()
        self.initialize_split_array(metadata_df, source_domain, target_domain)

        # Populate metadata fields
        self._metadata_fields = ["domain", "category", "y"]
        metadata_df = metadata_df[self._metadata_fields]
        possible_metadata_values = {
            "domain": DOMAIN_NET_DOMAINS,
            "category": DOMAIN_NET_CATEGORIES,
            "y": range(self._n_classes),
        }
        self._metadata_map, metadata = map_to_id_array(
            metadata_df, possible_metadata_values
        )
        self._metadata_array = torch.from_numpy(metadata.astype("long"))

        # Eval
        self.initialize_eval_grouper()
        super().__init__(root_dir, download, self._split_scheme)

    def get_input(self, idx) -> str:
        img_path = os.path.join(self.data_dir, self._input_image_paths[idx])
        img = Image.open(img_path).convert("RGB")
        return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
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
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric, self._eval_grouper, y_pred, y_true, metadata
        )

    def initialize_split_dicts(self):
        if self.split_scheme == "official":
            self._split_dict: Dict[str, int] = {
                "train": 0,
                "val": 1,
                "test": 2,
                "id_test": 3,
            }
            self._split_names: Dict[str, str] = {
                "train": "Train",
                "val": "Validation (OOD)",
                "test": "Test (OOD)",
                "id_test": "Test (ID)",
            }
            self._source_domain_splits = [0, 3]
        else:
            raise ValueError(f"Split scheme {self.split_scheme} is not recognized.")

    def initialize_split_array(self, metadata_df, source_domain, target_domain):
        def get_split(row):
            if row["domain"] == source_domain:
                if row["split"] == "train":
                    return 0
                elif row["split"] == "test":
                    return 3
            elif row["domain"] == target_domain:
                if row["split"] == "train":
                    return 1
                elif row["split"] == "test":
                    return 2
            else:
                raise ValueError(
                    f"Domain should be one of {source_domain}, {target_domain}"
                )

        self._split_array = metadata_df.apply(
            lambda row: get_split(row), axis=1
        ).to_numpy()

    def initialize_eval_grouper(self):
        if self.split_scheme == "official":
            self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=["category"]
            )
        else:
            raise ValueError(f"Split scheme {self.split_scheme} not recognized.")
