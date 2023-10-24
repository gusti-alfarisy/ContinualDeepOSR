import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

from custom_dataset import DataFrameImageDataset
from sklearn.model_selection import train_test_split

def uecfood100_datalaoder(ROOT, batch_size=32, img_size=224, num_workers=2, shuffle=True, validation=False, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_path = os.path.join(ROOT, "UECFOOD100")

    if validation:
        train_raw_df, val_raw_df, test_raw_df = preprocess_df_uecfood100_20(dataset_path, .8, .1, validation=validation)
        val_ds = DataFrameImageDataset(val_raw_df, transform=None, img_size=img_size)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        train_raw_df, test_raw_df = preprocess_df_uecfood100_20(dataset_path, .8, .1, validation=validation)


    train_ds = DataFrameImageDataset(train_raw_df, transform=None, img_size=img_size)
    test_ds = DataFrameImageDataset(test_raw_df, transform=None, img_size=img_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if validation:
        return train_dl, val_dl, test_dl
    else:
        return train_dl, test_dl

def preprocess_df_uecfood100_20(ds_path, train_split, val_split, seed=1, set_all_label_to=None, validation=False):
    # High samples at label: 1, 6, 12, 17, 23, 36, 68, 87
    # Above samples were deleted.
    file_paths = []
    labels = []
    class_list = os.listdir(ds_path)
    class_list = [int(x) for x in class_list if os.path.isdir(os.path.join(ds_path, x))]
    class_list.sort()
    # map_class_list = {idx: label for idx, label in enumerate(class_list)}

    # part -= 1
    # k_list = list(range(0 + (20 * part), 20 + (20 * part)))

    for idx, k in enumerate(class_list):
        label = k
        k_path = os.path.join(ds_path, str(label))
        file_list = os.listdir(k_path)
        for file in file_list:
            if file.split('.')[1] == 'txt': continue
            f_path = os.path.join(k_path, file)
            file_paths.append(f_path)
            labels.append(label)

    file_series = pd.Series(file_paths, name='file_paths')
    label_series = pd.Series(labels, name='labels')
    df = pd.concat([file_series, label_series], axis=1)

    unique_labels = df['labels'].unique()

    map_labels = {key: i for i, key in enumerate(unique_labels)}

    if set_all_label_to is not None:
        df['idx_label'] = set_all_label_to
    else:
        # Correcting the label to start at 0
        df['idx_label'] = df['labels'].apply(lambda x: map_labels[x])

    strat = df['labels']

    trainval_df, test_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=seed,
                                            stratify=strat)
    if validation:
        strat = trainval_df['labels']
        dsplit = val_split / train_split
        train_df, val_df = train_test_split(trainval_df, train_size=dsplit, shuffle=True, random_state=seed,
                                             stratify=strat)
        return train_df, val_df, test_df

    return trainval_df, test_df