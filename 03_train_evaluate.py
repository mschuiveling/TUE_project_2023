import fiftyone as fo
import fiftyone.zoo as foz
import torch, detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode
from fiftyone import ViewField as F
from detectron2.checkpoint import DetectionCheckpointer

from hooks import LossEvalHook
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import wandb

from detectron2.data.transforms import Transform
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from detectron2.data import transforms as T

# Load the datasets
train_dataset = fo.Dataset.from_dir(
    dataset_dir="/mnt/d/TIL_Melanoma_train_database/Manual_segmentation/tiles/coco/tiles/train/51_dataset",
    dataset_type=fo.types.FiftyOneDataset,
    label_field="ground_truth",
    tags="train",
)

test_dataset = fo.Dataset.from_dir(
    dataset_dir="/mnt/d/TIL_Melanoma_train_database/Manual_segmentation/tiles/coco/tiles/test/51_dataset",
    dataset_type=fo.types.FiftyOneDataset,
    label_field="ground_truth",
    tags="test",
)

train_view = train_dataset
test_view = test_dataset

from detectron2.structures import BoxMode
from fiftyone import ViewField as F
from detectron2.data import transforms as T
import torchvision.transforms as transforms
import PIL, cv2

def get_fiftyone_dicts(samples):
    samples.compute_metadata()

    dataset_dicts = []
    for sample in samples.select_fields(
        ["id", "filepath", "tags", "metadata", "detections", "ground_truth"]
    ):
        height = sample.metadata["height"]
        width = sample.metadata["width"]
        record = {}
        record["file_name"] = sample.filepath
        record["image_id"] = sample.id
        record["height"] = height
        record["width"] = width

        objs = []

        ground_truth = sample.ground_truth
        for det in ground_truth.detections:
            tlx, tly, w, h = det.bounding_box
            bbox = [
                int(tlx * width),
                int(tly * height),
                int(w * width),
                int(h * height),
            ]
            # Convert to COCO RLE
            fo_poly = det.to_polyline(tolerance=0)
            poly = [(x * width, y * height) for x, y in fo_poly.points[0]]
            poly = [p for x in poly for p in x]

            if len(poly) > 4:
                if det.label == "Tumor":
                    category_id = 0 # Assigning an integer value
                elif det.label == "Immune cells":
                    category_id = 1  # Assigning an integer value
                else:
                    category_id = 2  # Assigning an integer value

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": [poly],
                    "category_id": category_id,
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


DatasetCatalog.register(
    "train", lambda view=train_view: get_fiftyone_dicts(train_dataset)
)
DatasetCatalog.register("test", lambda view=test_view: get_fiftyone_dicts(test_dataset))

MetadataCatalog.get("train").set(
    thing_classes=["Tumor", "Immune cells", "Other"],
    thing_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
)
MetadataCatalog.get("test").set(
    thing_classes=["Tumor", "Immune cells", "Other"]
)
MetadataCatalog.get("test").set(
    thing_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
)

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
import copy
import torch
import numpy as np
from detectron2.structures import BitMasks
from pycocotools import mask as mask_utils


from hooks import LossEvalHook
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
from detectron2.data import build_detection_train_loader


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)
                ),
            ),
        )
        return hooks

import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
import wandb

# Wandb Login
wandb.login()
wandb.init(project="TIL_detectron2", sync_tensorboard=True, name='maskrcnn_50_incl_weights')


# Setup Configurations
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.TEST.EVAL_PERIOD = 100
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530] # RGB
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    return cfg

def train_model(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    cfg = setup_cfg()
    train_model(cfg)