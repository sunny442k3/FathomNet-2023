import os
import requests
from shutil import copyfileobj
import json
from multiprocessing.pool import ThreadPool


def read_json(filename):
    return json.load(open(filename, "r"))


def delete_empty_files(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)


def download_image(image):
    if not os.path.exists(image[0]):
        resp = requests.get(image[1], stream=True)
        resp.raw.decode_content = True
        with open(image[0], 'wb') as f:
            copyfileobj(resp.raw, f)

    
def train_download():
    delete_empty_files("./imgs/train/")
    train_json = read_json("./object_detection/train.json")
    train_imgs = [
        ["./imgs/train/"+i["file_name"], i["coco_url"]] for i in train_json["images"] if not os.path.exists("./imgs/train/"+i["file_name"])
    ]
    pool = ThreadPool(1000)
    results = pool.map(download_image, train_imgs)
    pool.close()
    pool.join()


def valid_download():
    delete_empty_files("./imgs/valid/")
    valid_json = read_json("./object_detection/eval.json")
    valid_imgs = [
        ["./imgs/valid/"+i["file_name"], i["coco_url"]] for i in valid_json["images"] if not os.path.exists("./imgs/valid/"+i["file_name"])
    ]
    pool = ThreadPool(1000)
    results = pool.map(download_image, valid_imgs)
    pool.close()
    pool.join()


if __name__ == "__main__":
    
    if not os.path.exists("./imgs"):
        os.mkdir("./imgs")
        os.mkdir("./imgs/train/")
        os.mkdir("./imgs/valid/")
    else:
        if not os.path.exists("./imgs/train/"):
            os.mkdir("./imgs/train/")

        if not os.path.exists("./imgs/valid/"):
            os.mkdir("./imgs/valid/")
    
    
    train_download()

    valid_download()