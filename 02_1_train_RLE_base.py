import os
import wandb
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import sys
sys.path.append('/home/mschuive/detectron2/detectron2_scripts')
from utilities import MyDatasetMapper, CocoAnnotationProcessor

# Set up the logger
setup_logger()

# Log in to Wandb
wandb.login()

# Define paths and directories
annotation_path = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/open_source_dataset_tiles_final/til_melanoma.json"
output_dir = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/open_source_dataset_tiles_final/"

# Define dataset_combinations dictionary
dataset_combinations = {
    # "fold_0": {"train": "train_dataset_stratified_fold_0", "val": "val_dataset_stratified_fold_0"},
    # "fold_1": {"train": "train_dataset_stratified_fold_1", "val": "val_dataset_stratified_fold_1"},
    # "fold_2": {"train": "train_dataset_stratified_fold_2", "val": "val_dataset_stratified_fold_2"},
    # "fold_3": {"train": "train_dataset_stratified_fold_3", "val": "val_dataset_stratified_fold_3"},
    "fold_4": {"train": "train_dataset_stratified_fold_4", "val": "val_dataset_stratified_fold_4"}
}

# Initialize the CocoAnnotationProcessor with dataset_combinations
processor = CocoAnnotationProcessor(annotation_path, output_dir)

# Load COCO annotations
processor.load_data()
processor.group_images_by_annotator()
processor.write_coco_fold_json()

# Register datasets in Detectron2
for fold in range(5):
    train_dataset_stratified_name = f"train_dataset_stratified_fold_{fold}"
    val_dataset_stratified_name = f"val_dataset_stratified_fold_{fold}"

    # Access temporary annotation file paths
    temp_train_annotation_path = os.path.join(output_dir, f"temp_train_annotations_stratified_fold_{fold}.json")
    temp_val_annotation_path = os.path.join(output_dir, f"temp_val_annotations_stratified_fold_{fold}.json")

    # Register the datasets using register_coco_instances
    register_coco_instances(
        train_dataset_stratified_name,  # Specify the desired train dataset name
        {},
        temp_train_annotation_path,
        "/path/to/your/images/"
    )
    register_coco_instances(
        val_dataset_stratified_name,  # Specify the desired val dataset name
        {},
        temp_val_annotation_path,
        "/path/to/your/images/"
    )

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
            max_dets_per_image=1000,
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
                brightness_max=1.1,
                flip_prob=0.5,
                blur_sigma=0.5,
                randomlight=0.5,        
                min_scale=0.8,
                max_scale=1.2,
                target_height=640,
                target_width=640,
            ),
        )

# WandB – Config is a variable that holds and saves hyperparameters and inputs
def sweep_config():
    sweep_config = {
        "method": "bayes",  # grid, random, bayes
        "metric": {"name": "segm/APm", "goal": "maximize"},
        "parameters": {
            # "BATCH_SIZE_PER_IMAGE": {"values": [512]},
            # "NUM_WORKERS": {"values": [4]},
            # "IMS_PER_BATCH": {"values": [2, 4, 6]},
            # "brightness_min": {"min": 0.3, "max": 1.0},
            # "brightness_max": {"min": 1.0, "max": 1.4},
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
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATALOADER.MASK_FORMAT = "rle"  # or "polygon" if you're using polygon encoding
    cfg.INPUT.MASK_FORMAT = ("bitmask")
    cfg.OUTPUT_DIR = f"/home/mschuive/detectron2/detectron2_scripts/all_classes/{wandb.run.name}{wandb.config.cross_val}"

    return cfg

def train_model():
    wandb.init(
        sync_tensorboard=True,
        name="TUE_BEP_1",
    )

    cfg = setup_cfg()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def wandb_sweep():
    config_s = sweep_config()
    sweep_id = wandb.sweep(config_s, project="TUE_BEP_1")
    wandb.agent(sweep_id, train_model)

if __name__ == "__main__":
    wandb_sweep()