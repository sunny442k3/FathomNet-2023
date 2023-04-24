import pandas as pd
import torch
from itertools import product
import numpy as np
from pathlib import Path
from fathomnet.api import images, boundingboxes
from fathomnet.scripts.fathomnet_generate import generate_coco_dataset
from fathomnet.models import GeoImageConstraints
from tqdm import tqdm
import ast
from PIL import ImageDraw
import random

def label_map(category, n_classes=290):
    category = ast.literal_eval(category)
    labels = [0]*n_classes
    for category_id in category:
        labels[int(category_id)] = 1
    return labels

def average_precision(output, target, k_max=20):
    pass

def mean_average_precision(output, target, k_max=20):
    pass

def get_data(concepts, max_depth=800):
    # get data from give concepts
    # concepts: string or list of concepts
    # max_depth: max depth of the image is taken
    image_dir = Path(f'./content/')
    if isinstance(concepts, str):
        image_dir = Path(f'./content/{label}')
        image_dir.mkdir(exist_ok=True, parents=True)
        gersemia_constraints = GeoImageConstraints(concept=concepts, maxDepth=max_depth)
        gersemia_images = images.find(gersemia_constraints)
        generate_coco_dataset(gersemia_images, image_dir)
    else:
        total_data = []
        for label in tqdm(concepts):
            image_dir.mkdir(exist_ok=True, parents=True)
            gersemia_constraints = GeoImageConstraints(concept=concepts[0], maxDepth=max_depth)
            gersemia_images = images.find(gersemia_constraints)
            total_data.extend(gersemia_images)
        generate_coco_dataset(total_data, image_dir)
            

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return 