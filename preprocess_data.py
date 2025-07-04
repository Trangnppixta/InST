import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from pathlib import Path
import random
import shutil
import pandas as pd

def copy_folder(input_dir, output_dir, sz):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

        with Image.open(input_path) as img:
            resized = img.resize(sz, Image.LANCZOS)
            resized.save(output_path)

def split_dataset(data_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    image_paths = list(data_dir.glob('*'))

    random.shuffle(image_paths)
    val_size = int(len(image_paths) * val_ratio)
    val_set = image_paths[:val_size]
    train_set = image_paths[val_size:]

    # Prepare output dirs
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for path in train_set:
        shutil.move(path, train_dir / path.name)
    for path in val_set:
        shutil.move(path, val_dir / path.name)

if __name__=='__main__':
    # data_dir = '/home/trangnguyenphuong/InST/data/pixta_data/2039131_4'
    # output_dir = '/home/trangnguyenphuong/InST/data/pixta_data/2039131_4'
    # sz = (512, 512)
    # copy_folder(input_dir, output_dir, sz)
    # split_dataset(data_dir, output_dir)

    data_dir = './data/pixta'
    output_dir = './data/pixta_data'
    style_csv = './data/updated_style.csv'
    style_df = pd.read_csv(style_csv)
    missing_items = 0

    for _, row in style_df.iterrows():
        item_id = row['id']
        style = row['style_id']
        filename = str(item_id) + '_1' + '.jpg'

        src_path = os.path.join(data_dir, filename)
        dst_dir = os.path.join(output_dir, style)
        dst_path = os.path.join(dst_dir, filename)

        if not os.path.exists(src_path):
            missing_items += 1
            continue

        os.makedirs(dst_dir, exist_ok=True)
        # Move or copy the image
        shutil.copy2(src_path, dst_path)

    print(missing_items)

