 # Create custom datasetmapper to apply data augmentations
from detectron2.data import transforms as T
import copy
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
import torch as torch
import numpy as np

class MyDatasetMapper(DatasetMapper):
    def __init__(
        self,
        cfg,
        is_train,
        brightness_min=0.85,  # aanpassing niet werkzaam, aanpassen hieronder (regel 314)
        brightness_max=1.0,
        flip_prob=0.5,
        blur_sigma=1.0,
        randomlight=0.5,
        min_scale=0.8,
        max_scale=1.2,
        target_height=640,
        target_width=640,
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
                T.ResizeScale(min_scale, max_scale, target_height, target_width),
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

from scipy.ndimage import gaussian_filter
from detectron2.data import transforms as T
import numpy as np

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
    
import json
import os
from sklearn.model_selection import StratifiedKFold

class CocoAnnotationProcessor:
    def __init__(self, annotation_path, output_dir, dataset_combinations=None):
        self.annotation_path = annotation_path
        self.output_dir = output_dir
        self.data = None
        self.images = None
        self.annotator_images = None
        self.dataset_combinations = dataset_combinations

    def load_data(self):
        with open(self.annotation_path, "r") as f:
            self.data = json.load(f)
            self.images = self.data['images']

    def group_images_by_annotator(self):
        self.annotator_images = {}
        for img in self.images:
            annotator = img['annotated_by']
            if annotator not in self.annotator_images:
                self.annotator_images[annotator] = []
            self.annotator_images[annotator].append(img)

    def write_coco_fold_json(self):
        annotation_dir = os.path.dirname(self.annotation_path)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx_total, val_idx_total) in enumerate(kf.split(self.images, [0]*len(self.images))):
            train_images = []
            val_images = []
            for annotator, images_list in self.annotator_images.items():
                train_images_annotator = [img for i, img in enumerate(images_list) if i in train_idx_total]
                val_images_annotator = [img for i, img in enumerate(images_list) if i in val_idx_total]
                train_images.extend(train_images_annotator)
                val_images.extend(val_images_annotator)

            train_data = {k: self.data[k] for k in ['info', 'categories']}
            val_data = {k: self.data[k] for k in ['info', 'categories']}

            train_data['images'] = train_images
            train_data['annotations'] = [anno for anno in self.data['annotations'] if anno['image_id'] in [img['id'] for img in train_images]]

            val_data['images'] = val_images
            val_data['annotations'] = [anno for anno in self.data['annotations'] if anno['image_id'] in [img['id'] for img in val_images]]

            temp_train_annotation_path = os.path.join(self.output_dir, f"temp_train_annotations_stratified_fold_{fold}.json")
            temp_val_annotation_path = os.path.join(self.output_dir, f"temp_val_annotations_stratified_fold_{fold}.json")

            with open(temp_train_annotation_path, "w") as f:
                json.dump(train_data, f)
            with open(temp_val_annotation_path, "w") as f:
                json.dump(val_data, f)

