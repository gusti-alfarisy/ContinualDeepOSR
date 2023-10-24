import os.path

import pandas as pd
from torch.utils.data import DataLoader

from custom_dataset import LatentDatasetTask, LatentDatasetTaskQCCPN, TaskDataLoaderHandler
from data.pretrained_model import PretrainedModelsFeature
import torch
import numpy as np
import torchvision.transforms.functional as Fvision
from imblearn.over_sampling import RandomOverSampler

from myutils import make_dir

data_config = {
    'oxfordpet': {
        'n_unknown': 7,
        'n_known_per_task': 5,
        'n_known': 30,
        'n_class': 37
    },
    # 93
    'uecfood100': {
        'n_unknown': 23,
        'n_known_per_task': 10,
        'n_known': 70,
        'n_class': 93
    }
}

def extract_feature(backbone, dl, out_path="", cuda=True):
    # import time
    def rotate_images(images, rotation_degree=90):
        im_1 = Fvision.rotate(images, rotation_degree)
        rotation_degree += rotation_degree
        im_2 = Fvision.rotate(images, rotation_degree)
        rotation_degree += rotation_degree
        im_3 = Fvision.rotate(images, rotation_degree)
        return im_1, im_2, im_3

    model = PretrainedModelsFeature(backbone)
    model = model.cuda() if cuda else model
    final_list = []
    final_list90 = []
    final_list180 = []
    final_list270 = []
    for (images, labels) in dl:
        images = images.cuda() if cuda else images
        labels = labels.cuda() if cuda else labels
        activations = model(images)
        activations_list = activations.detach().cpu().numpy()
        label_list = labels.detach().cpu().numpy()

        im_90, im_180, im_270 = rotate_images(images, 90)
        activations_list90 = model(im_90).detach().cpu().numpy()
        activations_list180 = model(im_180).detach().cpu().numpy()
        activations_list270 = model(im_270).detach().cpu().numpy()
        for act, lbl in zip(activations_list, label_list):
            final_list.append([act, lbl])
        for act, lbl in zip(activations_list90, label_list):
            final_list90.append([act, lbl])
        for act, lbl in zip(activations_list180, label_list):
            final_list180.append([act, lbl])
        for act, lbl in zip(activations_list270, label_list):
            final_list270.append([act, lbl])

    final_list = np.array(final_list, dtype=object)

    dir_out = os.path.dirname(out_path)
    firstname = os.path.basename(out_path).split("_")[0]
    with open(os.path.join(dir_out, f"{firstname}_{backbone}_seed1.npy"), "wb") as f:
        np.save(f, final_list)
    with open(os.path.join(dir_out, f"{firstname}_{backbone}_90deg_seed1.npy"), "wb") as f:
        np.save(f, final_list90)
    with open(os.path.join(dir_out, f"{firstname}_{backbone}_180deg_seed1.npy"), "wb") as f:
        np.save(f, final_list180)
    with open(os.path.join(dir_out, f"{firstname}_{backbone}_270deg_seed1.npy"), "wb") as f:
        np.save(f, final_list270)

def split_unknown(list_known_class, list_unknown_class, raw_data):
    latent_known = []
    label_known = []
    latent_unknown = []
    label_unknown = []
    map_idx_known = {e: i for i, e in enumerate(list_known_class)}
    map_idx_unknown = {e: i for i, e in enumerate(list_unknown_class, len(list_known_class))}

    print("map idx: ", map_idx_known)
    print("map idx unknown: ", map_idx_unknown)
    for d in raw_data:
        if int(d[1]) in list_known_class:
            latent_known.append(d[0])
            label_known.append(map_idx_known[d[1]])

        else:
            latent_unknown.append(d[0])
            label_unknown.append(map_idx_unknown[d[1]])


    return latent_known, label_known, latent_unknown, label_unknown

def load_datasets(dataset, backbone, verbose=0, with_degree=False, validation=False, seed=1, batch_size=32, non_continual=False, one_task=False, known_class=None, unknown_class=None):

    def get_task_dl(lat_data, lbl_data, lat90=None, lat180=None, lat270=None):
        prev = 0
        dl_task_list = []
        task_name_list = []
        if one_task:
            task = f"TASK: ({0}, {n_known_class})"
            if with_degree:
                ds_task = LatentDatasetTaskQCCPN(lat_data, lat90, lat180, lat270, lbl_data,
                                                 task_name=task)
            else:
                ds_task = LatentDatasetTask(lat_data, lbl_data, task_name=task)

            return TaskDataLoaderHandler([task], [ds_task], batch_size=batch_size)
        else:
            for x in range(0, n_known_class, n_class_per_task):
                task = f"TASK: ({x}, {x+n_class_per_task})"
                if non_continual:
                    filter_label = lbl_data < x+n_class_per_task
                else:
                    filter_label = np.logical_and(lbl_data >= x, lbl_data < x+n_class_per_task)

                latent_task = lat_data[filter_label]
                label_task = lbl_data[filter_label]

                if with_degree:
                    latent90_task = lat90[filter_label]
                    latent180_task = lat180[filter_label]
                    latent270_task = lat270[filter_label]
                    ds_task = LatentDatasetTaskQCCPN(latent_task, latent90_task, latent180_task, latent270_task, label_task,
                                                     task_name=task)
                else:
                    ds_task = LatentDatasetTask(latent_task, label_task, task_name=task)

                dl_task_list.append(ds_task)
                prev = x
                task_name_list.append(task)

            # print("task name list", task_name_list)

            return TaskDataLoaderHandler(task_name_list, dl_task_list, batch_size=batch_size)

    # n_inputs from mobilenet
    if backbone == 'mobilenet_v3_large':
        n_inputs = 960
    else:
        raise Exception("backbone unavailable")

    n_raw_class = data_config[dataset]['n_class']
    n_class_per_task = data_config[dataset]['n_known_per_task']
    n_known_class = data_config[dataset]['n_known']
    n_unknown_class = data_config[dataset]['n_unknown']

    if verbose:
        print("loading datasets with config")
        print(f"name: {dataset}")
        print(data_config[dataset])

    if known_class is None:
        list_known_class = list(range(n_known_class))
        list_unknown_class = list(range(n_known_class, n_raw_class))
    else:
        list_known_class = known_class
        list_unknown_class = unknown_class

    n_outputs = n_class_per_task
    if one_task:
        n_outputs = n_known_class

    if validation:
        train_raw = np.load(f'data/raw_latent/{dataset}/train_{backbone}_seed{seed}.npy', allow_pickle=True)
        val_raw = np.load(f'data/raw_latent/{dataset}/val_{backbone}_seed{seed}.npy', allow_pickle=True)

        if with_degree:
            train90_raw = np.load(f'data/raw_latent/{dataset}/train_{backbone}_90deg_seed{seed}.npy', allow_pickle=True)
            val90_raw = np.load(f'data/raw_latent/{dataset}/val_{backbone}_90deg_seed{seed}.npy', allow_pickle=True)

            train180_raw = np.load(f'data/raw_latent/{dataset}/train_{backbone}_180deg_seed{seed}.npy', allow_pickle=True)
            val180_raw = np.load(f'data/raw_latent/{dataset}/val_{backbone}_180deg_seed{seed}.npy', allow_pickle=True)

            train270_raw = np.load(f'data/raw_latent/{dataset}/train_{backbone}_270deg_seed{seed}.npy', allow_pickle=True)
            val270_raw = np.load(f'data/raw_latent/{dataset}/val_{backbone}_270deg_seed{seed}.npy', allow_pickle=True)

    else:
        train_raw = np.load(f'data/raw_latent/{dataset}/trainval_{backbone}_seed{seed}.npy', allow_pickle=True)
        if with_degree:
            train90_raw = np.load(f'data/raw_latent/{dataset}/trainval_{backbone}_90deg_seed{seed}.npy', allow_pickle=True)
            train180_raw = np.load(f'data/raw_latent/{dataset}/trainval_{backbone}_180deg_seed{seed}.npy', allow_pickle=True)
            train270_raw = np.load(f'data/raw_latent/{dataset}/trainval_{backbone}_270deg_seed{seed}.npy', allow_pickle=True)

    test_raw = np.load(f'data/raw_latent/{dataset}/test_{backbone}_seed{seed}.npy', allow_pickle=True)
    if with_degree:
        test90_raw = np.load(f'data/raw_latent/{dataset}/test_{backbone}_90deg_seed{seed}.npy', allow_pickle=True)
        test180_raw = np.load(f'data/raw_latent/{dataset}/test_{backbone}_180deg_seed{seed}.npy', allow_pickle=True)
        test270_raw = np.load(f'data/raw_latent/{dataset}/test_{backbone}_270deg_seed{seed}.npy', allow_pickle=True)

    # Train and val data
    lat_train_known, lbl_train_known, lat_unknown, lbl_unknown = split_unknown(list_known_class, list_unknown_class, train_raw)
    if with_degree:
        lat90_train_known, lbl90_train_known, lat90_unknown, lbl90_unknown = split_unknown(list_known_class, list_unknown_class, train90_raw)
        lat180_train_known, lbl180_train_known, lat180_unknown, lbl180_unknown = split_unknown(list_known_class, list_unknown_class, train180_raw)
        lat270_train_known, lbl270_train_known, lat270_unknown, lbl270_unknown = split_unknown(list_known_class, list_unknown_class, train270_raw)

    if validation:
        lat_val_known, lbl_val_known, lat2_unknown, lbl2_unknown = split_unknown(list_known_class, list_unknown_class, val_raw)
        lat_unknown = lat_unknown + lat2_unknown
        lbl_unknown = lbl_unknown + lbl2_unknown

        if with_degree:
            lat90_val_known, lbl90_val_known, lat90_unknown, lbl90_unknown = split_unknown(list_known_class, list_unknown_class,
                                                                                               val90_raw)
            lat180_val_known, lbl180_val_known, lat180_unknown, lbl180_unknown = split_unknown(list_known_class, list_unknown_class,
                                                                                                   val180_raw)
            lat270_val_known, lbl270_val_known, lat270_unknown, lbl270_unknown = split_unknown(list_known_class, list_unknown_class,
                                                                                                   val270_raw)

    # Testing data
    lat_test_known, lbl_test_known, lat2_unknown, lbl2_unknown = split_unknown(list_known_class, list_unknown_class, test_raw)
    lat_unknown = lat_unknown + lat2_unknown
    lbl_unknown = lbl_unknown + lbl2_unknown


    if with_degree:
        lat90_test_known, lbl90_test_known, lat90_test_unknown, lbl90_unknown = split_unknown(list_known_class, list_unknown_class,
                                                                                           test90_raw)
        lat90_unknown = lat90_unknown + lat90_test_unknown
        lat180_test_known, lbl180_test_known, lat180_test_unknown, lbl180_unknown = split_unknown(list_known_class, list_unknown_class,
                                                                                               test180_raw)
        lat180_unknown = lat180_unknown + lat180_test_unknown
        lat270_test_known, lbl270_test_known, lat270_test_unknown, lbl270_unknown = split_unknown(list_known_class, list_unknown_class,
                                                                                               test270_raw)
        lat270_unknown = lat270_unknown + lat270_test_unknown

    lat_train_known = np.array(lat_train_known)
    lbl_train_known = np.array(lbl_train_known)

    print("BEFORE over sampling", lat_train_known.shape) if verbose else None
    if with_degree:
        len_fea = len(lat_train_known[0])
        latconcat = np.concatenate((lat_train_known, lat90_train_known, lat180_train_known, lat270_train_known), 1)
        ros = RandomOverSampler(random_state=seed)
        latconcat_known, lbl_train_known = ros.fit_resample(latconcat, lbl_train_known)
        lat_train_known = latconcat_known[:, :len_fea]
        lat90_train_known = latconcat_known[:, len_fea:len_fea*2]
        lat180_train_known = latconcat_known[:, len_fea*2:len_fea*3]
        lat270_train_known = latconcat_known[:, len_fea*3:]
    else:
        ros = RandomOverSampler(random_state=seed)
        lat_train_known, lbl_train_known = ros.fit_resample(lat_train_known, lbl_train_known)
    print("AFTER over sampling", lat_train_known.shape) if verbose else None

    lat_test_known = np.array(lat_test_known)
    lbl_test_known = np.array(lbl_test_known)
    if validation:
        lat_val_known = np.array(lat_val_known)
        lbl_val_known = np.array(lbl_val_known)


    if with_degree:
        lat90_train_known = np.array(lat90_train_known)
        lat180_train_known = np.array(lat180_train_known)
        lat270_train_known = np.array(lat270_train_known)

        lat90_test_known = np.array(lat90_test_known)
        lat180_test_known = np.array(lat180_test_known)
        lat270_test_known = np.array(lat270_test_known)

        if validation:
            lat90_val_known = np.array(lat90_val_known)
            lat180_val_known = np.array(lat180_val_known)
            lat270_val_known = np.array(lat270_val_known)
    else:
        lat90_train_known, lat180_train_known, lat270_train_known = None, None, None
        lat90_test_known, lat180_test_known, lat270_test_known = None, None, None
        lat90_val_known, lat180_val_known, lat270_val_known = None, None, None

    tdl_train = get_task_dl(lat_train_known, lbl_train_known, lat90=lat90_train_known, lat180=lat180_train_known, lat270=lat270_train_known)
    tdl_test = get_task_dl(lat_test_known, lbl_test_known, lat90=lat90_test_known, lat180=lat180_test_known, lat270=lat270_test_known)
    if validation:
        tdl_val = get_task_dl(lat_val_known, lbl_val_known, lat90=lat90_val_known, lat180=lat180_val_known, lat270=lat270_val_known)

    # Unknown part
    if with_degree:
        ds_u = LatentDatasetTaskQCCPN(lat_unknown, lat90_unknown, lat180_unknown, lat270_unknown, lbl_unknown, task_name='open-set')
    else:
        ds_u = LatentDatasetTask(lat_unknown, lbl_unknown, task_name='open-set')

    dl_u = DataLoader(ds_u, batch_size=batch_size, shuffle=True, num_workers=2)
    # TODO think to returning train with different degree
    if validation:
        if with_degree:
            return tdl_train, tdl_test, tdl_val, n_inputs, n_outputs, dl_u, lat_train_known, lat90_train_known, lat180_train_known, lat270_train_known, lbl_train_known
        else:
            return tdl_train, tdl_test, tdl_val, n_inputs, n_outputs, dl_u, lat_train_known, lbl_train_known

    if with_degree:
        return tdl_train, tdl_test, n_inputs, n_outputs, dl_u, lat_train_known, lat90_train_known, lat180_train_known, lat270_train_known, lbl_train_known
    return tdl_train, tdl_test, n_inputs, n_outputs, dl_u, lat_train_known, lbl_train_known


def train_class_distribution(dataset, seed=1, validation=False):
    # Oxfordpet 50
    import matplotlib.pyplot as plt
    import seaborn as sns
    if validation:
        datapath = os.path.join("data", "raw_latent", dataset, f"train_mobilenet_v3_large_seed{seed}.npy")
    else:
        datapath = os.path.join("data", "raw_latent", dataset, f"trainval_mobilenet_v3_large_seed{seed}.npy")

    raw_data = np.load(datapath, allow_pickle=True)
    n_known_class = data_config[dataset]['n_known']
    list_known_class = list(range(n_known_class))
    lat_train_known, lbl_train_known, lat_unknown, lbl_unknown = split_unknown(list_known_class, raw_data)
    lbl_train_known = pd.Series(np.array(lbl_train_known), name="Class Index")
    sns.displot(lbl_train_known, binwidth=1)
    if validation:
        save_path = os.path.join("_result", "class_distribution", f"{dataset}_train.png")
    else:
        save_path = os.path.join("_result", "class_distribution", f"{dataset}_trainval.png")

    make_dir(save_path)
    plt.savefig(save_path)
    plt.show()