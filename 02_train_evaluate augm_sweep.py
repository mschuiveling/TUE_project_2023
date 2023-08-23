import fiftyone as fo
import fiftyone.zoo as foz
import torch
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os
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
from detectron2.structures import BoxMode
from fiftyone import ViewField as F
from detectron2.data import transforms as T
import torchvision.transforms as transforms
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
import copy
import torch
import numpy as np
from detectron2.structures import BitMasks
from pycocotools import mask as mask_utils


# Wandb Login
wandb.login()


from detectron2.data.transforms import Transform
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from detectron2.data import transforms as T

# Create a custom transform class (Gaussian Blur) as this is not included in detectron2 
class GaussianBlurTransform(T.Transform):
    """
    Transforms pixel values using Gaussian Blur.
    """

    def __init__(self, sigma: float):
        """
        Args:
            sigma (float): Std deviation of the kernel
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply gaussian transform on the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, gaussian blur would not
                require interpolation.
        Returns:
            ndarray: blurred image(s).
        """
        # Apply gaussian_filter to each channel independently
        if img.ndim == 2:
            img[:, :] = gaussian_filter(img[:, :], self.sigma, mode="mirror")
        elif img.ndim == 3:
            img = gaussian_filter(
                img, sigma=(self.sigma, self.sigma, 0), mode="mirror"
            )  # sigma 0 for last dimension so that blurring doesn't happen across channels
        elif img.ndim == 4:
            img = gaussian_filter(
                img, sigma=(0, self.sigma, self.sigma, 0), mode="mirror"
            )
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        The inverse is a no-op.
        """
        return T.NoOpTransform()
class RandomGaussianBlur(T.Augmentation):
    """
    Apply Gaussian Blur by choosing a random sigma
    """

    def __init__(self, sigma, sample_style="range"):
        """
        Args:
            sigma (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the sigma.
                If ``sample_style=="choice"``, a list of sigmas to sample from
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self._init(locals())

    def get_transform(self, image):
        if self.is_range:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = np.random.choice(self.sigma)

        return GaussianBlurTransform(sigma)

# Load the datasets
train_dataset = fo.Dataset.from_dir(
    dataset_dir="/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/train/51_dataset",
    dataset_type=fo.types.FiftyOneDataset,
    label_field="ground_truth",
    tags="train",
)
test_dataset = fo.Dataset.from_dir(
    dataset_dir="/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/test/51_dataset",
    dataset_type=fo.types.FiftyOneDataset,
    label_field="ground_truth",
    tags="test",
)

train_view = train_dataset
test_view = test_dataset

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
                    category_id = 0  # Assigning an integer value
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
    thing_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
)
MetadataCatalog.get("test").set(thing_classes=["Tumor", "Immune cells", "Other"])
MetadataCatalog.get("test").set(thing_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])

# Create custom datasetmapper to apply data augmentations
class MyDatasetMapper(DatasetMapper):
    def __init__(
        self,
        cfg,
        is_train=True,
        brightness_min=0.85,
        brightness_max=1.0,
        flip_prob=0.5,
        blur_sigma=1.0,
        randomlight=0.2,
        # RandomExtent_min = [0.8, 1.2],
        # RandomExtent_max = [0.8, 1.2],
        # RandomExtent = ([0.8, 1.2], [0.8, 1.2])
    ):
        super().__init__(cfg, is_train, instance_mask_format="polygon")
        self.augmentations = T.AugmentationList(
            [
                T.RandomBrightness(brightness_min, brightness_max),
                T.RandomFlip(prob=flip_prob),
                T.RandomCrop("absolute", (640, 640)),
                T.RandomLighting(randomlight),
                RandomGaussianBlur(blur_sigma),
                T.RandomExtent([0.8, 1.2], [0.8, 1.2]),
                # Add other transformations here
            ]
        )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


from hooks import LossEvalHook
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
from detectron2.data import build_detection_train_loader

os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '15'  # number of failed runs to accept 

# Create custom trainer to add custom evaluator and wandb sweep on image augmentations
class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name, cfg, True, output_folder, max_dets_per_image=3000
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)
                ),
            ),
        )
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=MyDatasetMapper(
                cfg,
                is_train=True,
                brightness_min=0.5,
                # brightness_max=wandb.config.brightness_max,
                brightness_max=1.1,
                flip_prob=0.5,
                blur_sigma=0.5,
                randomlight=0.5,
            )
        )


import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
import wandb


# WandB – Config is a variable that holds and saves hyperparameters and inputs
def sweep_config():
    sweep_config = {
        "method": "bayes",  # grid, random, bayes
        "metric": {"name": "validation_loss", "goal": "minimize"},
        "parameters": {
            # "BASE_LR": {"values": [0.001]}, # recheck with larger iterations and batch size
            # "MAX_ITER": {"values": [5000]},
            # "BATCH_SIZE_PER_IMAGE": {"values": [512]},
            # "NUM_WORKERS": {"values": [4]},
            # "IMS_PER_BATCH": {"values": [2, 4, 6]},
            # "brightness_min": {"min": 0.3, "max": 1.0},
            # "brightness_max": {"min": 1.0, "max": 1.4},
            # "flip_prob": {"min": 0.1, "max": 1.0},
            # "blur_sigma": {"min": 0.1, "max": 1.6},
            # "randomlight": {"min": 0.1, "max": 1.0},
            "anch_1": {"min": 8, "max": 16},
            "anch_2": {"min": 16, "max": 32},
            "anch_3": {"min": 32, "max": 128},
            "anch_4": {"min": 64, "max": 256},
            "anch_5": {"min": 128, "max": 512},
            "model": {"values": [
                # "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
                                "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                                # "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml", # best working backbone for now 
                                   ]},
            }
    }
    return sweep_config

# Setup Configurations
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            wandb.config.model
        )
    )
    cfg.DATASETS.TRAIN = ("train",)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[wandb.config.anch_1, wandb.config.anch_2, wandb.config.anch_3, wandb.config.anch_4, wandb.config.anch_5]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8 , 16, 32, 64, 128]]
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        wandb.config.model
    )    
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.TEST.EVAL_PERIOD = 100
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    return cfg

def train_model():
    wandb.init(
         sync_tensorboard=True,
        name="anchor_boxes_sweep",
    )

    cfg= setup_cfg()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def wandb_sweep():
    config_s = sweep_config()
    sweep_id = wandb.sweep(config_s, project="TIL_detectron2")
    wandb.agent(sweep_id, train_model)

if __name__ == "__main__":

    wandb_sweep()
