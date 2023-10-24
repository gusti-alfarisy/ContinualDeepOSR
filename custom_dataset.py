import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import time

class DataFrameImageDataset(Dataset):

    def __init__(self, dataframe, transform=None, img_size=224):
        """

        :param dataframe: consist of dataframe with two columns, path images (including root) and label
        """
        self.df = dataframe
        self.prop_path = "file_paths"
        self.prop_label = "labels"
        self.prop_idx_label = "idx_label"

        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((img_size, img_size)),
                torchvision.transforms.ToTensor()
            ])

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # time.sleep(0.1)
        row = self.df.iloc[idx]
        path = row[self.prop_path]
        idx_label = row[self.prop_idx_label]
        pil_image = Image.open(path).convert('RGB')
        return self.transform(pil_image), idx_label

    def get_labels(self):
        return self.df[self.prop_idx_label]

class LatentDatasetTask(Dataset):
    def __init__(self, latent, labels, task_name):
        """
        :param dataframe: consist of dataframe with two columns, path images (including root) and label
        """
        self.latent = latent
        self.labels = labels
        self.task_name = task_name
    def __len__(self):
        return len(self.latent)
    def __getitem__(self, idx):
        # time.sleep(0.1)
        return self.latent[idx], self.labels[idx], self.task_name

class LatentDatasetTaskQCCPN(Dataset):
    def __init__(self, latent, laten90, latent180, latent270, labels, task_name):
        """
        :param dataframe: consist of dataframe with two columns, path images (including root) and label
        """
        self.latent = latent
        self.latent90 = laten90
        self.latent180 = latent180
        self.latent270 = latent270
        self.labels = labels
        self.task_name = task_name

    def __len__(self):
        return len(self.latent)

    def __getitem__(self, idx):
        # time.sleep(0.1)
        return self.latent[idx], self.latent90[idx], self.latent180[idx], self.latent270[idx], self.labels[idx], self.task_name


class TaskDataLoaderHandler:
    def __init__(self, task_list=[], lds_list=[], batch_size=32):
        self.task_name = task_list
        self.lds_list = lds_list
        self.dl_list = [DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2) for ds in self.lds_list]
        self.key_mask = list(range(len(self)))
        self.map_tasks_ds = {k:v for k, v in zip(task_list, self.lds_list)}
        self.map_tasks_dl = {k:v for k, v in zip(task_list, self.dl_list)}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.dl_list[self.key_mask[key]]

        else:
            return self.map_tasks_dl[key]

    def dataset(self, key):
        if isinstance(key, int):
            return self.lds_list[self.key_mask[key]]
        else:
            return self.map_tasks_ds[key]

    def permute_tasks(self):
        self.key_mask = torch.randperm(len(self)).tolist()

    def get_task_name(self, idx):
        return self.task_name[self.key_mask[idx]]

    def __len__(self):
        return len(self.task_name)


class SupervisedDataset(Dataset):
    def __init__(self, images=[], labels=[]):
        """
        :param images: -
        :param labels: -
        """
        self.images = []
        self.labels = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def add(self, image, label):
        self.images.append(image)
        self.labels.append(label)