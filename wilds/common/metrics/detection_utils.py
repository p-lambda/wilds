import datetime
import os

import numpy as np
import tqdm

from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


CATEGORIES = [
    {"supercategory": "human",         "id": 1, "name": "pedestrian"},
    {"supercategory": "vehicle",       "id": 2, "name": "car"},
    {"supercategory": "bike",          "id": 3, "name": "bicycle"},
    {"supercategory": "traffic light", "id": 4, "name": "traffic light"},
    {"supercategory": "traffic sign",  "id": 5, "name": "traffic sign"},
]

NAME_MAPPING = {
    "bike": "bicycle",
    "caravan": "car",
    "motor": "motorcycle",
    "person": "pedestrian",
    "van": "car",
}

IGNORE_MAP = {
    'rider': 'pedestrian',
    'truck': 'car',
    'bus': 'car',
    'train': 'car',
    'motorcycle': 'bicycle',
    "other person": "pedestrian",
    "other vehicle": "car",
    "trailer": 'car',
}


class COCOV2(COCO):
    """Modify the COCO API to support annotations dictionary as input."""

    def __init__(self, annotations):
        """Init."""
        super().__init__()
        # initialize the annotations in COCO format without saving as json.
        assert isinstance(
            annotations, dict
        ), "annotation file format {} not supported".format(
            type(annotations)
        )
        self.dataset = annotations
        self.createIndex()


def evaluate_det(labels, preds, separate_classes=False):
    """Load the ground truth and prediction results."""
    # Convert the annotation file to COCO format
    ann_coco = bdd100k2coco_det(labels)
    coco_gt = COCOV2(ann_coco)

    # Load results and convert the predictions
    pred_res = convert_preds(preds, ann_coco)
    coco_dt = coco_gt.loadRes(pred_res)

    cat_ids = coco_dt.getCatIds()
    cat_names = [cat["name"] for cat in coco_dt.loadCats(cat_ids)]

    img_ids = sorted(coco_gt.getImgIds())
    ann_type = "bbox"
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = img_ids

    return evaluate_workflow(coco_eval, cat_ids, cat_names,
                             separate_classes=separate_classes)


def bdd100k2coco_det(labels):
    """Converting BDD100K Detection Set to COCO format."""
    coco, cat_name2id = init()
    coco["type"] = "instances"
    image_id, ann_id = 1, 1

    for frame in tqdm.tqdm(labels):
        image = dict()
        set_image_attributes(image, frame['name'], image_id)
        coco["images"].append(image)

        if frame['labels'] is None:
            continue
        for label in frame['labels']:
            if label['box2d'] is None:
                continue

            category_ignored, category_id = process_category(
                label['category'], cat_name2id
            )
            if category_ignored:
                continue

            annotation = dict(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                bdd100k_id=str(label['id']),
            )
            set_object_attributes(annotation, label, category_ignored)
            set_box_object_geometry(annotation, label)
            coco["annotations"].append(annotation)

            ann_id += 1
        image_id += 1

    return coco


def init():
    """Initialize the annotation dictionary."""
    coco = defaultdict(list)
    coco["categories"] = CATEGORIES
    coco["categories"] += [
    ]
    category_name2id = {
        category["name"]: category["id"] for category in coco["categories"]
    }

    return coco, category_name2id


def set_image_attributes(image, image_name, image_id, video_name=""):
    """Set attributes for the image dict."""
    image.update(
        dict(
            file_name=os.path.join(video_name, image_name),
            height=720,
            width=1280,
            id=image_id,
        )
    )


def process_category(category_name, cat_name2id):
    """Check whether the category should be ignored and get its ID."""
    category_name = NAME_MAPPING.get(category_name, category_name)
    if category_name not in cat_name2id:
        category_name = IGNORE_MAP[category_name]
        category_ignored = True
    else:
        category_ignored = False
    category_id = cat_name2id[category_name]
    return category_ignored, category_id


def set_object_attributes(annotation, label, ignore):
    """Set attributes for the ann dict."""
    attributes = label['attributes']
    if attributes is None:
        return
    iscrowd = bool(attributes.get("crowd", False))
    annotation.update(
        dict(
            iscrowd=int(iscrowd or ignore),
            ignore=int(ignore),
        )
    )


def set_box_object_geometry(annotation, label):
    """Parsing bbox, area, polygon for bbox ann."""
    box_2d = label['box2d']
    if box_2d is None:
        return
    x1 = box_2d['x1']
    y1 = box_2d['y1']
    x2 = box_2d['x2']
    y2 = box_2d['y2']

    annotation.update(
        dict(
            bbox=[x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            area=float((x2 - x1 + 1) * (y2 - y1 + 1)),
            segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
        )
    )


def convert_preds(res, ann_coco, max_det=100):
    """Convert the prediction into the coco eval format."""
    res = pred_to_coco(res, ann_coco)

    # get the list of image_ids in res.
    name = "image_id"
    image_idss = set()
    for item in res:
        if item[name] not in image_idss:
            image_idss.add(item[name])
    image_ids = sorted(list(image_idss))

    # sort res by 'image_id'.
    res = sorted(res, key=lambda k: int(k["image_id"]))

    # get the start and end index in res for each image.
    image_id = image_ids[0]
    idx = 0
    start_end = {}
    for i, res_i in enumerate(res):
        if i == len(res) - 1:
            start_end[image_id] = (idx, i + 1)
        if res_i[name] != image_id:
            start_end[image_id] = (idx, i)
            idx = i
            image_id = res_i[name]

    # cut number of detections to max_det for each image.
    res_max_det = []
    more_than_max_det = 0
    for image_id in image_ids:
        r_img = res[start_end[image_id][0] : start_end[image_id][1]]
        if len(r_img) > max_det:
            more_than_max_det += 1
            r_img = sorted(
                r_img, key=lambda k: float(k["score"]), reverse=True
            )[:max_det]
        res_max_det.extend(r_img)

    if more_than_max_det > 0:
        print(
            "Some images have more than {0} detections. Results were "
            "cut to {0} detections per images on {1} images.".format(
                max_det, more_than_max_det
            )
        )

    return res_max_det


def pred_to_coco(pred, ann_coco):
    """Convert the predictions into a compatible format with COCOAPIs."""
    # update the prediction results
    imgs_maps = {
        os.path.split(item["file_name"])[-1]: item["id"]
        for item in ann_coco["images"]
    }
    cls_maps = {item["name"]: item["id"] for item in ann_coco["categories"]}

    # backward compatible replacement
    naming_replacement_dict = {
        "person": "pedestrian",
        "motor": "motorcycle",
        "bike": "bicycle",
    }
    for p in pred:
        # add image_id and category_id
        cls_name = p["category"]
        if cls_name in naming_replacement_dict.keys():
            cls_name = naming_replacement_dict[cls_name]
        p["category_id"] = cls_maps[cls_name]
        p["image_id"] = imgs_maps[p["name"]]
        x1, y1, x2, y2 = p["bbox"]  # x1, y1, x2, y2
        p["bbox"] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

    return pred


def evaluate_workflow(coco_eval, cat_ids, cat_names, separate_classes=False):
    n_tit = 12  # number of evaluation titles
    n_cls = len(cat_ids)  # 10/8 classes for BDD100K detection/tracking
    n_thr = 10  # [.5:.05:.95] T=10 IoU thresholds for evaluation
    n_rec = 101  # [0:.01:1] R=101 recall thresholds for evaluation
    n_area = 4  # A=4 object area ranges for evaluation
    n_mdet = 3  # [1 10 100] M=3 thresholds on max detections per image

    eval_param = {
        "params": {
            "imgIds": [],
            "catIds": [],
            "iouThrs": np.linspace(
                0.5,
                0.95,
                int(np.round((0.95 - 0.5) / 0.05) + 1),
                endpoint=True,
            ).tolist(),
            "recThrs": np.linspace(
                0.0,
                1.00,
                int(np.round((1.00 - 0.0) / 0.01) + 1),
                endpoint=True,
            ).tolist(),
            "maxDets": [1, 10, 100],
            "areaRng": [
                [0 ** 2, 1e5 ** 2],
                [0 ** 2, 32 ** 2],
                [32 ** 2, 96 ** 2],
                [96 ** 2, 1e5 ** 2],
            ],
            "useSegm": 0,
            "useCats": 1,
        },
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ],
        "counts": [n_thr, n_rec, n_cls, n_area, n_mdet],
        "precision": -np.ones(
            (n_thr, n_rec, n_cls, n_area, n_mdet), order="F"
        ),
        "recall": -np.ones((n_thr, n_cls, n_area, n_mdet), order="F"),
    }
    stats_all = -np.ones((n_cls, n_tit))

    for i, (cat_id, cat_name) in enumerate(zip(cat_ids, cat_names)):
        print("\nEvaluate category: %s" % cat_name)
        coco_eval.params.catIds = [cat_id]
        # coco_eval.params.useSegm = ann_type == "segm"
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_all[i, :] = coco_eval.stats
        eval_param["precision"][:, :, i, :, :] = coco_eval.eval[
            "precision"
        ].reshape((n_thr, n_rec, n_area, n_mdet))
        eval_param["recall"][:, i, :, :] = coco_eval.eval["recall"].reshape(
            (n_thr, n_area, n_mdet)
        )

    if separate_classes:
        return stats_all[:, 0]

    # Print evaluation results
    stats = np.zeros((n_tit, 1))
    print("\nOverall performance")
    coco_eval.eval = eval_param
    coco_eval.summarize()

    for i in range(n_tit):
        column = stats_all[:, i]
        if len(column > -1) == 0:
            stats[i] = -1
        else:
            stats[i] = np.mean(column[column > -1], axis=0)
    return stats[0]
