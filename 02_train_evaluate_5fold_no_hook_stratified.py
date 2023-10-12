import torch
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_test_loader, DatasetCatalog, build_detection_train_loader
from hooks import LossEvalHook
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import wandb
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy
import torchvision.transforms.functional as F
from scipy.ndimage.filters import gaussian_filter
from detectron2.data.datasets import register_coco_instances
from sklearn.model_selection import KFold, cross_val_score

os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "15"  # number of failed runs to accept
wandb.login()

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

import os
import json
from sklearn.model_selection import KFold
from detectron2.data.datasets import register_coco_instances
import json
import os
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load the COCO annotations
annotation_path = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/open_source_dataset_tiles_final/til_melanoma.json" # path to annotation file from 01_coco_creator_5fold

with open(annotation_path, "r") as f:
    data = json.load(f)
    images = data['images']
# Group images by the 'annotated_by' field
annotator_images = {}
for img in images:
    annotator = img['annotated_by']
    if annotator not in annotator_images:
        annotator_images[annotator] = []
    annotator_images[annotator].append(img)

print('annotations done by',annotator_images.keys(), 'in this dataset')

# Directory where the original annotation resides
annotation_dir = os.path.dirname(annotation_path)

# 5-Fold cross-validation for each annotator
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the 5 folds
for fold, (train_idx_total, val_idx_total) in enumerate(kf.split(images, [0]*len(images))):
    train_images = []
    val_images = []
    for annotator, images_list in annotator_images.items():
        train_images_annotator = [img for i, img in enumerate(images_list) if i in train_idx_total]
        val_images_annotator = [img for i, img in enumerate(images_list) if i in val_idx_total]
        train_images.extend(train_images_annotator)
        val_images.extend(val_images_annotator)


    # Create temporary annotation files for training and validation
    train_data = {k: data[k] for k in ['info', 'categories']}
    val_data = {k: data[k] for k in ['info', 'categories']}

    train_data['images'] = train_images
    train_data['annotations'] = [anno for anno in data['annotations'] if anno['image_id'] in [img['id'] for img in train_images]]

    val_data['images'] = val_images
    val_data['annotations'] = [anno for anno in data['annotations'] if anno['image_id'] in [img['id'] for img in val_images]]

    with open(os.path.join(annotation_dir, f"temp_train_annotations_stratified_fold_{fold}.json"), "w") as f:
        json.dump(train_data, f)
    with open(os.path.join(annotation_dir, f"temp_val_annotations_stratified_fold_{fold}.json"), "w") as f:
        json.dump(val_data, f)

    # Register datasets in Detectron2
    train_dataset_stratified_name = f"train_dataset_stratified_fold_{fold}"
    val_dataset_stratified_name = f"val_dataset_stratified_fold_{fold}"

    # Check if the dataset is already registered and unregister it
    if train_dataset_stratified_name in DatasetCatalog.list():
        DatasetCatalog.remove(train_dataset_stratified_name)
    if val_dataset_stratified_name in DatasetCatalog.list():
        DatasetCatalog.remove(val_dataset_stratified_name)

    # Register the datasets
    register_coco_instances(train_dataset_stratified_name, {}, os.path.join(annotation_dir, f"temp_train_annotations_stratified_fold_{fold}.json"), "/path/to/your/images/")
    register_coco_instances(val_dataset_stratified_name, {}, os.path.join(annotation_dir, f"temp_val_annotations_stratified_fold_{fold}.json"), "/path/to/your/images/")

# Define the dataset_combinations dictionary
dataset_combinations = {
    # "fold_0": {"train": "train_dataset_stratified_fold_0", "val": "val_dataset_stratified_fold_0"},
    # "fold_1": {"train": "train_dataset_stratified_fold_1", "val": "val_dataset_stratified_fold_1"},
    # "fold_2": {"train": "train_dataset_stratified_fold_2", "val": "val_dataset_stratified_fold_2"},
    # "fold_3": {"train": "train_dataset_stratified_fold_3", "val": "val_dataset_stratified_fold_3"},
    "fold_4": {"train": "train_dataset_stratified_fold_4", "val": "val_dataset_stratified_fold_4"}
}


# Create custom datasetmapper to apply data augmentations
class MyDatasetMapper(DatasetMapper):
    def __init__(
        self,
        cfg,
        is_train=True,
        brightness_min=0.85,  # aanpassing niet werkzaam, aanpassen hieronder (regel 314)
        brightness_max=1.0,
        flip_prob=0.5,
        blur_sigma=1.0,
        randomlight=0.5,
    ):
        super().__init__(cfg, is_train, instance_mask_format="bitmask")
        self.augmentations = T.AugmentationList(
            [
                T.RandomBrightness(brightness_min, brightness_max),
                T.RandomFlip(prob=flip_prob),
                T.RandomCrop("absolute", (640, 640)),
                T.RandomLighting(randomlight),
                RandomGaussianBlur(blur_sigma),
                T.RandomExtent([0.8, 1.2], [0.8, 1.2]),
                # T.ResizeScale(min_scale=0.1, max_scale=2.0, target_height=1024, target_width=1024),
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
            dataset_name=dataset_name,
            tasks = ["bbox", "segm"],
            distributed=True,
            output_dir=output_folder,
            max_dets_per_image=3000,
            allow_cached_coco=False,
            )

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=MyDatasetMapper(
                cfg,
                is_train=True,
                brightness_min=0.5,
                # brightness_max=wandb.config.brightness_max,
                brightness_max=1.1,
                flip_prob=0.5,
                blur_sigma=0.5,
                randomlight=0.5,
            ),
        )

    # Hierboven data augmentations toevoegen


# WandB – Config is a variable that holds and saves hyperparameters and inputs
def sweep_config():
    sweep_config = {
        "method": "bayes",  # grid, random, bayes
        "metric": {"name": "segm/APm", "goal": "maximize"},
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
            # "anch_1": {"min": 8, "max": 16},
            # "anch_2": {"min": 16, "max": 32},
            # "anch_3": {"min": 32, "max": 128},
            # "anch_4": {"min": 64, "max": 256},
            # "anch_5": {"min": 128, "max": 512},
            # "NMS_THRESH_TEST" : {"values": [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]},
            # "SCORE_THRESH_TEST" : {"min": 0.01, "max": 0.30},
            "cross_val": {"values": list(dataset_combinations.keys())},
            "model": {
                "values": [
                    # "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
                    # "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
                    # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
                    # "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
                    # "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",  # this one for TUE experiments
                    # "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
                    # "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
                    # "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                    # "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml", # best working backbone for now
                ]
            },
        },
    }
    return sweep_config


# Setup Configurations
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(wandb.config.model))
    # Fetch the dataset names based on the selected combination from wandb config
    combo_key = wandb.config.cross_val
    train_dataset_stratified_name = dataset_combinations[combo_key]['train']
    val_dataset_stratified_name = dataset_combinations[combo_key]['val']
    cfg.DATASETS.TRAIN = (train_dataset_stratified_name,)
    cfg.DATASETS.TEST = (val_dataset_stratified_name,)

        # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
    #     [
    #         wandb.config.anch_1,
    #         wandb.config.anch_2,
    #         wandb.config.anch_3,
    #         wandb.config.anch_4,
    #         wandb.config.anch_5,
    #     ]
    # ]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8 , 16, 32, 64, 128]]
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(wandb.config.model)
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.TEST.EVAL_PERIOD = 100
    cfg.TEST.DETECTIONS_PER_IMAGE = 2500
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = wandb.config.NMS_THRESH_TEST

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATALOADER.MASK_FORMAT = "rle"  # or "polygon" if you're using polygon encoding
    cfg.INPUT.MASK_FORMAT = ("bitmask")
    cfg.OUTPUT_DIR = f"/mnt/d/TIL_Melanoma_train_database/cell_segmentation/open_source_dataset_tiles_final/output/{wandb.run.name}"

    return cfg

def train_model():
    wandb.init(
        sync_tensorboard=True,
        name="TUE",
    )

    cfg = setup_cfg()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def wandb_sweep():
    config_s = sweep_config()
    sweep_id = wandb.sweep(config_s, project="TUE_2 evaluate cross_val")
    wandb.agent(sweep_id, train_model)

if __name__ == "__main__":
    wandb_sweep()