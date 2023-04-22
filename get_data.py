import pandas as pd
from tqdm import tqdm
from pathlib import Path
from fathomnet.api import images, boundingboxes
from fathomnet.scripts.fathomnet_generate import generate_coco_dataset
from fathomnet.models import GeoImageConstraints

category_csv = pd.read_csv(category_key_file)
len_labels = []
image_paths = []
labels = category_csv['name'].tolist()
for label in tqdm(labels):
    image_dir = Path(f'./content/{label}')
    image_dir.mkdir(exist_ok=True, parents=True)
    gersemia_constraints = GeoImageConstraints(concept=label)
    gersemia_images = images.find(gersemia_constraints)
    generate_coco_dataset(gersemia_images, image_dir)
