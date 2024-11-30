from typing import Dict, List, Optional, NamedTuple, Tuple
import os
import torch
from torch.utils.data import Dataset, random_split
import cv2
from math import sqrt
import numpy as np
from PIL import Image, ImageFile
import random
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


ConversionTable = Dict[str, int]
ImagePath = str
Detections = List[int]


def list_files(directories: List[str]):
    files = [
        f"{directory}/" + f
        for directory in directories
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    return files


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    cv2.resize(image, dim, interpolation=inter, dst=image)

    # return the resized image
    return image


class DatasetLoaderClassification:

    def __init__(
        self,
        split: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        seed: int = 69,
    ) -> None:
        if len(split) != 3:
            raise ValueError("Split should have 3 values!")

        self.n_classes: int = 0

        self.seed = seed
        self.split: Tuple[float, float, float] = self._norm_split(split)
        self.images_entries: List[str] = []
        self.train_entries: List[str] = []
        self.val_entries: List[str] = []

    def _norm_split(
        self, split: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        total = sum(split)

        l = list(split)

        l[0] = split[0] / total
        l[1] = split[1] / total
        l[2] = split[2] / total

        return tuple(l)

    def _split(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.train_entries, self.val_entries, self.test_entries = random_split(
            self.images_entries, self.split, generator=generator
        )

    def _get_file_path(self, path: str, file_map: Optional[Dict[str, str]]):
        if file_map is None:
            return path

        result = file_map.get(path, "")

        if not result:
            raise ValueError(
                f"The file {path} was not found in any provided directories!"
            )

        return result

    def from_directory(self, directory: str):
        files = list_files([directory])
        self.images_entries = files
        self._split()
        return self

    def __len__(self):
        return len(self.images_entries)

    def train(self, transform=None):
        return ECGDataset(self.train_entries, transform)

    def validation(self, transform=None):
        return ECGDataset(self.val_entries, transform)

    def test(self, transform=None):
        return ECGDataset(self.test_entries, transform)

    def whole(self, transform=None):
        return ECGDataset(self.images_entries, transform)


class ECGDataset(Dataset):
    def __init__(self, data: List[str], transform):
        self.data = data
        self.rot_matrixes = torch.nn.functional.one_hot(torch.arange(0, 4), 4).float()
        self.rot_values = [-90, 0, 90, 180]
        self.transform = transform
        random.seed = time.time()

    def count(self):
        sm = np.zeros((self.n_classes))
        for i in range(self.__len__()):
            _, label = self.__getitem__(i)
            sm += label.numpy()
        return sm

    def mean(self) -> List[float]:
        r_sm: float = 0
        g_sm: float = 0
        b_sm: float = 0
        r_elements: int = 0
        g_elements: int = 0
        b_elements: int = 0

        for i in range(self.__len__()):
            img, _ = self.__getitem__(i)
            r, g, b = img[0], img[1], img[2]

            r_sm += r.sum().item()
            b_sm += g.sum().item()
            g_sm += b.sum().item()

            r_elements += r.numel()
            g_elements += g.numel()
            b_elements += b.numel()
        return (r_sm / r_elements, g_sm / g_elements, b_sm / b_elements)

    def std(self, mean=None) -> List[float]:
        if mean is None:
            r_mean, g_mean, b_mean = self.mean()
        else:
            r_mean, g_mean, b_mean = mean

        r_sm: float = 0
        g_sm: float = 0
        b_sm: float = 0

        r_elements: int = 0
        g_elements: int = 0
        b_elements: int = 0

        for i in range(self.__len__()):
            img, _ = self.__getitem__(i)
            img = np.array(img)
            img = img.reshape(3, *img.shape[:-1])
            r, g, b = img[0], img[1], img[2]

            r_sm += ((r - r_mean) ** 2).sum().item()
            g_sm += ((g - g_mean) ** 2).sum().item()
            b_sm += ((b - b_mean) ** 2).sum().item()

            r_elements += r.numel()
            g_elements += g.numel()
            b_elements += b.numel()

        return (
            sqrt(r_sm / r_elements),
            sqrt(g_sm / g_elements),
            sqrt(b_sm / b_elements),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image_filepath = self.data[idx]
        img = Image.open(image_filepath)

        k = random.randint(0, len(self.rot_values) - 1)
        rot = self.rot_matrixes[k]
        img = img.rotate(self.rot_values[k])

        return (self.transform(img), rot)
