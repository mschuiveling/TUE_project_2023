import os
import cv2
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
import fiftyone as fo
from pycocotools import mask as maskUtils
import numpy as np
import json

# Setup Detectron2 logger
setup_logger()

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = '/home/mschuive/detectron2/output/model_final.pth'

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50  # Set a custom testing threshold for non-maxium suppression to counter double detections of cells
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8 , 16, 32, 64, 128]]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    return cfg

import json
import os
import cv2
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
import fiftyone as fo

def detectron_to_fo(outputs, img_w, img_h):
    # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    detections = []
    instances = outputs["instances"].to("cpu")
    labels = ["Tumor", "Immune cells", "Other"]
    for pred_box, score, c, mask in zip(
        instances.pred_boxes, instances.scores, instances.pred_classes, instances.pred_masks,
    ):
        x1, y1, x2, y2 = pred_box
        
        fo_mask = mask.numpy()[int(y1):int(y2), int(x1):int(x2)]
        bbox = [float(x1) / img_w, float(y1) / img_h, float(x2 - x1) / img_w, float(y2 - y1) / img_h]
        detection = fo.Detection(label=labels[c], confidence=float(score), bounding_box=bbox, mask=fo_mask)
        detections.append(detection)
        
    return fo.Detections(detections=detections)

def run_predictions_on_folder(predictor, folder_path):
    # List all image files in the folder
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png'))]

    # Prepare dataset_dicts for predictions
    dataset_dicts = []
    for image_file in image_files:
        img = cv2.imread(image_file)
        height, width, _ = img.shape
        record = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": height,
            "width": width
        }
        dataset_dicts.append(record)

    # Run predictions and store results in a dictionary
    predictions = {}
    for d in dataset_dicts:
        img_w = d["width"]
        img_h = d["height"]
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        detections = detectron_to_fo(outputs, img_w, img_h)
        predictions[d["image_id"]] = detections

    return predictions

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

def detections_to_geojson(detections, image_path):
    # Convert fiftyone.Detections to a GeoJSON-like format compatible with QuPath
    # Create a GeoJSON-like dictionary for each detection
    features = []
    for detection in detections.detections:
        coordinates = detection.to_polyline(tolerance=0)
        fo_poly = detection.to_polyline(tolerance=0)
        for label in detection.label:
            if label == 'Tumor':
                detection.color = [200, 0, 0]
            elif label == 'Immune_cell':
                detection.color = [128, 0, 128]
            else: detection.color = [255, 200, 0]

        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        if fo_poly.points:
             poly = [(round(x * img_w), round(y * img_h)) for x, y in fo_poly.points[0]]
        feature = {
            "type": "Feature", 
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly[:] + [poly[0]]],  # Close the polygon
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": detection.label,
                    "color": detection.color,
                },
            },
           
        }
        features.append(feature)
        
    # Create a FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Save to a GeoJSON file
    image_filename = os.path.join(new_folder_path, os.path.basename(image_path))
    output_filename = image_filename.replace(".png", ".geojson")

    with open(output_filename, "w") as f:
        json.dump(feature_collection, f)


if __name__ == "__main__":
    # Load the trained model and setup predictor
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    # Path to the new folder containing images for prediction
    new_folder_path = '/mnt/d/TIL_Melanoma_train_database/cell_segmentation/tiles_1024/detectron2_inference/attempt_3_smaller_anchor'

    # Run predictions on the new folder
    predictions = run_predictions_on_folder(predictor, new_folder_path)

    # Export detections to GeoJSON for each image in the folder
    for image_id, detections in predictions.items():
        image_path = os.path.join(new_folder_path, image_id)
        detections_to_geojson(detections, image_path)
        print(f"Detections exported to {image_id.replace('.png', '.geojson')}")
    print ("Done!")

    # Now, you have exported the detections to GeoJSON files for each image in the folder.
    # You can find the GeoJSON files in the same folder as the input images.