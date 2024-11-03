import os

import torch

"""
install detectron2:
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
"""
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import numpy as np
import scipy

from detectron2 import model_zoo


def setup_mask_rcnn():
    cfg = get_cfg()
    # Load a pre-trained model from Detectron2's model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def perform_segmentation_and_show_all_classes(cfg, predictor, image_path):
    """Predict and show all classes above threshold"""
    im = Image.open(image_path).convert("RGB")
    np_image = np.array(im)

    # Perform inference
    outputs = predictor(np_image)
    # print(outputs)
    # Visualize the prediction
    v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Display the image
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()


def perform_segmentation(cfg, predictor, image_path, threshold_prob=0.5):
    """Only shows region and mask for best scoring class"""
    im = Image.open(image_path).convert("RGB")
    np_image = np.array(im)

    # Perform inference
    outputs = predictor(np_image)

    # Extract instances from prediction outputs
    instances = outputs["instances"]

    max_scoring_indices = [info[1] for info in sorted([(instances.scores[i].item(), i)
                                                       for i in range(len(instances.pred_classes))
                                                       if instances.scores[i].item() >= threshold_prob])]

    # Filter instances to keep only those with max scores per class
    filtered_instances = instances[max_scoring_indices]
    # print(filtered_instances)

    # Visualize the prediction with only the instances having the highest scores per class
    v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(filtered_instances)

    # Display the image
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    cfg, predictor = setup_mask_rcnn()
    image_path = "example.jpg"
    perform_segmentation(cfg, predictor, image_path)
