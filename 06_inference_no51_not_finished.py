import os
import cv2
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
import json
import numpy as np

# Setup Detectron2 logger
setup_logger()


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = "/home/mschuive/detectron2/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    return cfg

def mask_local_to_global(mask_coords, bbox_coords):
    """
    Convert mask coordinates from local (bounding box) to global (whole image) coordinates.

    Parameters:
        mask_coords (list of tuples): Mask coordinates in local (bounding box) coordinates.
        bbox_coords (list): Bounding box coordinates [x1, y1, x2, y2] in whole image coordinates.

    Returns:
        list of tuples: Mask coordinates in global (whole image) coordinates.
    """
    x_offset, y_offset = bbox_coords[:2]
    global_mask_coords = [(x + x_offset, y + y_offset) for x, y in mask_coords]
    return global_mask_coords


def detectron_to_dict(outputs, img_w, img_h):
    detections = []
    instances = outputs["instances"].to("cpu")
    labels = ["Tumor", "Immune cells", "Other"]
    for pred_box, score, c, mask in zip(
        instances.pred_boxes,
        instances.scores,
        instances.pred_classes,
        instances.pred_masks,
    ):
        x1, y1, x2, y2 = pred_box
        bbox = [
            float(x1) / img_w,
            float(y1) / img_h,
            float(x2 - x1) / img_w,
            float(y2 - y1) / img_h,
        ]

        detection = {
            "label": labels[c],
            "confidence": float(score),
            "bounding_box": bbox,
            "mask": mask.numpy().tolist(),

        }
        print (detection)
        detections.append(detection)
    return detections


def run_predictions_on_folder(predictor, folder_path):
    image_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith((".jpg", ".png"))
    ]
    predictions = {}
    for image_file in image_files:
        img = cv2.imread(image_file)
        height, width, _ = img.shape
        outputs = predictor(img)
        detections = detectron_to_dict(outputs, width, height)
        predictions[os.path.basename(image_file)] = detections
    return predictions


def detections_to_geojson(detections, image_path):
    features = []
    for detection in detections:
        bbox = detection["bounding_box"]
        poly = [  # Convert bounding box to polygon
            [bbox[0], bbox[1]],
            [bbox[0] + bbox[2], bbox[1]],
            [bbox[0] + bbox[2], bbox[1] + bbox[3]],
            [bbox[0], bbox[1] + bbox[3]],
            [bbox[0], bbox[1]],
        ]
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [poly]},
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": detection["label"],
                },
            },
        }
        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_filename = os.path.join(
        new_folder_path, os.path.basename(image_path).replace(".png", ".geojson")
    )
    with open(output_filename, "w") as f:
        json.dump(feature_collection, f)


if __name__ == "__main__":
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    new_folder_path = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/tiles_1024/detectron2_inference/attempt_3_smaller_anchor/new_code"
    predictions = run_predictions_on_folder(predictor, new_folder_path)
    for image_id, detections in predictions.items():
        image_path = os.path.join(new_folder_path, image_id)
        detections_to_geojson(detections, image_path)
        print(f"Detections exported to {image_id.replace('.png', '.geojson')}")
    print("Done!")
