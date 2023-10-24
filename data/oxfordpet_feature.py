import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

from custom_dataset import DataFrameImageDataset
from sklearn.model_selection import train_test_split


def oxfordpet_dataloader(ROOT, batch_size=32, img_size=224, num_workers=2, shuffle=True, validation=False, seed=1):
    # validation false is for val+train
    torch.manual_seed(seed)
    np.random.seed(seed)

    if validation:
        train_raw_df, val_raw_df, test_raw_df = preprocess_df_oxfordpet(ROOT, 0.9, validation=validation, seed=seed)
        val_ds = DataFrameImageDataset(val_raw_df, transform=None, img_size=img_size)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        train_raw_df, test_raw_df = preprocess_df_oxfordpet(ROOT, validation=validation, seed=seed)

    train_ds = DataFrameImageDataset(train_raw_df, transform=None, img_size=img_size)
    test_ds = DataFrameImageDataset(test_raw_df, transform=None, img_size=img_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if validation:
        return train_dl, val_dl, test_dl
    else:
        return train_dl, test_dl

def preprocess_df_oxfordpet(ROOT, train_split=0.9, validation=True, seed=123):
    dataset_path = os.path.join(ROOT, "Oxford_IIIT_Pet")
    im_paths = os.path.join(dataset_path, "images")
    test_annot_path = os.path.join(dataset_path, "annotations", "test.txt")
    trainval_annot_path = os.path.join(dataset_path, "annotations", "trainval.txt")

    # Preprocess test dataset
    test_dict = {"file_paths": [], "idx_label": []}
    with open(test_annot_path, "r") as f:
        lines = f.read().split('\n')
        for l in lines:
            data = l.split(" ")
            if len(data) < 2:
                continue
            test_dict["file_paths"].append(os.path.join(im_paths, f"{data[0]}.jpg"))
            test_dict["idx_label"].append(int(data[1]) - 1)

    test_df = pd.DataFrame(test_dict)

    trainval_dict = {"file_paths": [], "idx_label": []}
    with open(trainval_annot_path, "r") as f:
        lines = f.read().split('\n')
        for l in lines:
            data = l.split(" ")
            if len(data) < 2:
                continue
            trainval_dict["file_paths"].append(os.path.join(im_paths, f"{data[0]}.jpg"))
            trainval_dict["idx_label"].append(int(data[1]) - 1)

    trainval_df = pd.DataFrame(trainval_dict)

    strat = trainval_df['idx_label']

    if validation:
        train_df, val_df = train_test_split(trainval_df, train_size=train_split, shuffle=True, random_state=random_seed,
                                            stratify=strat)
        return train_df, val_df, test_df
    else:
        return trainval_df, test_df