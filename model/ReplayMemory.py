import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from custom_dataset import LatentDatasetTask, TaskDataLoaderHandler, LatentDatasetTaskQCCPN
import itertools

class MemoryScheme:
    def __init__(self, device, lat_train, lbl_train, type='dynamic', per_class=50, is_shuffle=True):
        if type is not 'dynamic':
            raise Exception("fixed or other type of memory is not implemented")
        self.memory_data = {}
        concatlist = [(lat, lbl) for lat, lbl in zip(lat_train, lbl_train)]
        random.shuffle(concatlist)
        for d in concatlist:
            if d[1] not in self.memory_data:
                self.memory_data[d[1]] = []
            if len(self.memory_data[d[1]]) < per_class:
                self.memory_data[d[1]].append(d[0])

        self.is_shuffle = is_shuffle
        self.device = device
        self.temp_memory_dl = None
        self.iter_memory = None

    def adjust(self, list_class, batch_size):
        latent = []
        labels = []
        for key, value in self.memory_data.items():
            if key in list_class:
                for data in self.memory_data[key]:
                    latent.append(data)
                    labels.append(key)

        ds = LatentDatasetTask(latent, labels, f"({list_class[0], list_class[-1]})")
        self.iter_memory = itertools.cycle(iter(DataLoader(ds, batch_size=batch_size, shuffle=self.is_shuffle, num_workers=2)))

    def take(self):
        return next(self.iter_memory)


class MemorySchemeQCCPN:
    def __init__(self, device, lat_train, lat90_train, lat180_train, lat270_train, lbl_train, type='dynamic', per_class=50, is_shuffle=True):
        if type is not 'dynamic':
            raise Exception("fixed or other type of memory is not implemented")
        self.memory_data = {}
        concatlist = [(lat, lat90, lat180, lat270, lbl) for lat, lat90, lat180, lat270, lbl in zip(lat_train, lat90_train, lat180_train, lat270_train, lbl_train)]
        random.shuffle(concatlist)
        for d in concatlist:
            if d[-1] not in self.memory_data:
                self.memory_data[d[-1]] = []
            if len(self.memory_data[d[-1]]) < per_class:
                self.memory_data[d[-1]].append((d[0], d[1], d[2], d[3]))

        self.is_shuffle = is_shuffle
        self.device = device
        self.temp_memory_dl = None
        self.iter_memory = None

    def adjust(self, list_class, batch_size):
        latent = []
        latent90 = []
        latent180 = []
        latent270 = []
        labels = []
        for key, value in self.memory_data.items():
            if key in list_class:
                for lat, lat90, lat180, lat270 in self.memory_data[key]:
                    latent.append(lat)
                    latent90.append(lat90)
                    latent180.append(lat180)
                    latent270.append(lat270)
                    labels.append(key)

        ds = LatentDatasetTaskQCCPN(latent, latent90, latent180, latent270, labels, f"({list_class[0], list_class[-1]})")
        self.iter_memory = itertools.cycle(
            iter(DataLoader(ds, batch_size=batch_size, shuffle=self.is_shuffle, num_workers=2)))

    def take(self):
        return next(self.iter_memory)
