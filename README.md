# TUE_project_2023
TUE BEP project 

# Detectron2 Training with W&B Integration

This repository contains a Python script for training machine learning models using the Detectron2 framework with the integration of Weights & Biases (W&B) for experiment tracking and hyperparameter tuning.

## Overview of 02_1_train_RLE_base.py

The script performs the following functions:

1. Sets up logging using Detectron2's logger.
2. Authenticates the user with W&B.
3. Processes COCO-style annotations for a dataset and writes them into separate JSON files for training and validation.
4. Registers the processed datasets with Detectron2.
5. Defines a custom trainer class that integrates with W&B and adds a custom evaluation method.
6. Sets up a W&B sweep configuration for hyperparameter tuning.
7. Configures the Detectron2 model and training parameters.
8. Initializes a W&B run and starts the training process.
9. Executes a W&B sweep to find the best model and hyperparameters.

## The `utilities.py` script includes the following key components:
- `MyDatasetMapper`: A custom dataset mapper that applies a variety of data augmentations such as random brightness changes, flips, crops, lighting adjustments, Gaussian blur, and resizing.
- `GaussianBlurTransform` and `RandomGaussianBlur`: Custom transformation classes for applying Gaussian blur as an augmentation during training.
- `CocoAnnotationProcessor`: A utility class for processing COCO-style annotations, grouping images by annotator, and writing stratified folds for cross-validation.

## COCODatasetGenerator
The `COCODatasetGenerator` script processes GeoJSON files and generates a COCO-style dataset JSON file. It supports the conversion of polygon annotations to the COCO format, including category assignment and bounding box calculation. The generated JSON file can then be used for training models in Detectron2.

## Visualize.ipynb
    This code load the annotations, the output direction for the saved images and the name of the test set from which the images are visualized
    # TUE Project 2023 - Visualize Inference Results

    This Jupyter Notebook is used to visualize the inference results of a Detectron2 model on a test set of images. The notebook assumes that the model has already been trained and the inference results have been saved to a JSON file.

    The notebook contains the following cells:

    - **Cell 0:** This markdown cell that describes the purpose and contents of the notebook.
    - **Cell 1:** Imports the necessary libraries and loads the annotations and test set information.
    - **Cell 2:** Visualizes random images from the test set with ground truth annotations.
    - **Cell 3:** Visualizes the predictions and saves the images to the output directory.
    - **Cell 4:** Creates dictionaries for visualizing false positives, false negatives, and class false positives.
    - **Cell 5:** Computes the IoU between each ground truth box and predicted boxes and creates dictionaries for true positives, false positives, false negatives, and class false positives.
    - **Cell 6:** Saves the visualizations of false positives, false negatives, and class false positives to the output directory.

    To use this notebook, make sure that the necessary libraries are installed and that the paths to the annotations, test set, and output directory are correct. Then, run each cell in order to visualize the inference results and create the necessary dictionaries for further analysis.


## Prerequisites

Before running the script, ensure you have the following:

- Python 3.6 or later.
- Detectron2 installed and configured properly.
- Weights & Biases account and CLI installed.
- COCO-style annotated dataset placed in the specified directory.

## Installation

Clone this repository to your local machine using:

```bash
git clone 'https://github.com/mschuiveling/TUE_project_2023'
```

Configuration
You can configure the script by editing the following variables:

annotation_path: Path to the COCO-style annotation file.
output_dir: Directory where the processed annotation files will be stored.
dataset_combinations: Dictionary defining the dataset splits.
The configuration for the model and the W&B sweep are set up in the sweep_config() and setup_cfg() functions, respectively.

Custom Training and Evaluation
The script includes a custom trainer MyTrainer which integrates with W&B and adds a custom evaluator for the COCO dataset.

W&B Sweep
The W&B sweep is configured to use Bayesian optimization to find the best hyperparameters for the model. The sweep configuration can be adjusted as needed.

Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Acknowledgments
Detectron2 contributors and maintainers.
The W&B team for providing an excellent tool for ML experiment tracking.