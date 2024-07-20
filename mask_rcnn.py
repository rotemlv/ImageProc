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


def perform_segmentation(cfg, predictor, image_path):
    im = Image.open(image_path).convert("RGB")
    np_image = np.array(im)

    # Perform inference
    outputs = predictor(np_image)

    # Visualize the prediction
    v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Display the image
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    cfg, predictor = setup_mask_rcnn()
    image_path = "example.jpg"
    perform_segmentation(cfg, predictor, image_path)
