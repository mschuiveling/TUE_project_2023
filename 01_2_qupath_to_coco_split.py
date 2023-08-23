import json
import glob
import os
from PIL import Image
import numpy as np
from pycocotools import coco
import cv2
from pathlib import Path

class COCODatasetGenerator:
    def __init__(self, geojson_path, output_file_name):
        self.geojson_path = geojson_path
        self.output_file_name = output_file_name
        self.image_nr = 0
        self.images = []
        self.annotations = []

    def generate_dataset(self):
        info = {
            "year": 2023,
            "version": "3",
            "description": "COCO_database_melanoma,_data_4_session",
            "contributor": "M. Schuiveling",
            "url": "none",
            "date_created": "07/04/2023",
        }
        

        # Load image categories
        categories = [
            {"supercategory": "cells", "id": 1, "name": "Immune cells"},
            {"supercategory": "cells", "id": 2, "name": "Tumor"},
            {"supercategory": "cells", "id": 3, "name": "Histiocyte"},
            {"supercategory": "cells", "id": 4, "name": "Endothelium"},
            {"supercategory": "cells", "id": 5, "name": "Melanophage"},
            {"supercategory": "cells", "id": 6, "name": "Neutrophil"},
            {"supercategory": "cells", "id": 7, "name": "Plasma cell"},
            {"supercategory": "cells", "id": 8, "name": "Eosinophil"},
            {"supercategory": "cells", "id": 9, "name": "Other"},
        ]

        # Iterate over the JSON files in the folder
        for geojson_name in glob.glob(os.path.join(self.geojson_path, "*.geojson")):
            self.image_nr += 1

            # Extract image information and Load image information for coco database
            self.images.append(
                {
                    "id": self.image_nr,
                    # "width": Image.open(geojson_name.replace(".geojson", ".png")).size,
                    # "height": Image.open(geojson_name.replace(".geojson", ".png")).size,
                    "width": 1024,
                    "height": 1024,
                    "file_name": str(geojson_name.replace(".geojson", ".png")),
                }
            )

            # Load the JSON data from file
            with open(geojson_name) as f:
                geojson_data = json.load(f)


            # Load annotations
            for feature in geojson_data["features"]:
                segmentation = feature["geometry"]["coordinates"]
                # GEOJSON also holds multipolygon which does not have a classification and holds list[list[int]] segmentation which gives errors
                if feature["geometry"]["type"] == "Polygon" and len(segmentation[0]) > 8:
                    segmentation_flt_1 = [
                        [list(map(float, inner_list)) for inner_list in outer_list]
                        for outer_list in segmentation
                    ]
                    segmentation_flt_2 = segmentation_flt_1[0]
                    segmentation_flt_3 = []
                    for sublist in segmentation_flt_2:
                        for element in sublist:
                            segmentation_flt_3.append(element)

                    classification = feature["properties"].get("classification", {})
                    # Map the category name to its ID, default is 9 (other)
                    category_id = {
                        "Immune cells": 1,
                        "Tumor": 2,
                        # "Histiocyte": 3,
                        # "Endothelium": 4,
                        # "Melanophage": 5,
                        # "Neutrophil": 6,
                        # "Plasma cell": 7,
                        # "Eosinophil": 8 
                        #  for now all other cells are classified as other to see whether training is at all possible with this dataset
                    }.get(classification.get("name", ""), 9)

                    # Create bbox coordinates
                    x = [point[0] for point in segmentation[0]]
                    y = [point[1] for point in segmentation[0]]
                    xmin = min(x)
                    xmax = max(x)
                    ymin = min(y)
                    ymax = max(y)
                    bbox = float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)
                    if xmax-xmin < 0.1:
                        print (geojson_name)
                    


                    # create larger bbox to take more context

                    self.annotations.append(
                        {
                            "image_id": self.image_nr,
                            "category_id": category_id,
                            "bbox": bbox,
                            "bbox_mode": 0,
                            "segmentation": [segmentation_flt_3],
                        }
                    )

        coco_data = {
            "info": info,
            "categories": categories,
            "images": self.images,
            "annotations": self.annotations,
        }

        # Convert the output to a text file
        print (self.output_file_name, 'COCO database has been formed' )
        with open(os.path.join(self.geojson_path, self.output_file_name), "w") as f:
            f.write(json.dumps(coco_data))


# Define the path to the GeoJSON files and the output file name
GEOJSON_path = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/train"
output_file_name = "coco_dataset_melanoma_cells_train.json"
dataset_generator = COCODatasetGenerator(GEOJSON_path, output_file_name)
dataset_generator.generate_dataset()

# Define the path to the GeoJSON files and the output file name
GEOJSON_path = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/test"
output_file_name = "coco_dataset_melanoma_cells_test.json"
dataset_generator = COCODatasetGenerator(GEOJSON_path, output_file_name)
dataset_generator.generate_dataset()


import fiftyone as fo
from fiftyone.types.dataset_types import COCODetectionDataset
   
til_dataset_train = fo.Dataset.from_dir(
    dataset_type = COCODetectionDataset,
    labels_path=  "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/train/coco_dataset_melanoma_cells_train.json"
    )

print(til_dataset_train)
til_dataset_train.merge_labels('segmentations', 'ground_truth')

export_dir = '/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/train/51_dataset'

label_field = "ground_truth"  # for example
dataset_type = fo.types.COCODetectionDataset  # for example

# Export the dataset
til_dataset_train.export(
    export_dir=export_dir,
    dataset_type=fo.types.FiftyOneDataset,
    label_field=label_field,
)

import fiftyone as fo
from fiftyone.types.dataset_types import COCODetectionDataset
   
til_dataset_test = fo.Dataset.from_dir(
    dataset_type = COCODetectionDataset,
    labels_path= "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/test/coco_dataset_melanoma_cells_test.json"
    )

print(til_dataset_test)
til_dataset_test.merge_labels('segmentations', 'ground_truth')

export_dir = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/coco_database_train_test/test/51_dataset"

label_field = "ground_truth"  # for example
dataset_type = fo.types.COCODetectionDataset  # for example

# Export the dataset
til_dataset_test.export(
    export_dir=export_dir,
    dataset_type=fo.types.FiftyOneDataset,
    label_field=label_field,
)