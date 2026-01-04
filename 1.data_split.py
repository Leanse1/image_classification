## added None class with images from roboflow (human,cycle,garage)

import json
import os
import glob
import shutil
from collections import Counter
import random

BASE_DIR = "/home/leanse/AI/interview/clearquote/exercise_1"
output_dir = "/home/leanse/AI/interview/clearquote"
data_dir = "/home/leanse/AI/interview/clearquote/data"
new_output_dir = "/home/leanse/AI/interview/clearquote/dataset"

train_split = 0.8
val_split = 0.1
test_split = 0.1

classes = os.listdir(data_dir)

for cls in classes:
    class_path = os.path.join(data_dir, cls)
    if not os.path.isdir(class_path):
        continue  # skip files if any
    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    train_end = int(len(images) * train_split)
    val_end = train_end + int(len(images) * val_split)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split_name, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_dir = os.path.join(new_output_dir, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

print("Dataset successfully split into train/val/test!")