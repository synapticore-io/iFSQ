import os.path as osp
import random
from glob import glob
from torchvision import transforms
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pickle
from PIL import Image
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import (
    Lambda,
    Compose,
    Resize,
    RandomCrop,
    CenterCrop,
    ToTensor,
)
import torch
import os


def get_transform(
    resolution,
    interpolation=InterpolationMode.BILINEAR,
    augment=None,
    ra_magnitude=9,
):

    transforms = [T.RandomHorizontalFlip(0.5)]

    if augment is not None:
        if augment == "ra":
            transforms.extend(
                [
                    T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude),
                ]
            )
        elif augment == "ta_wide":
            transforms.extend(
                [
                    T.TrivialAugmentWide(interpolation=interpolation),
                ]
            )

    transforms.extend(
        [
            ToTensor(),
            Resize(resolution),
            RandomCrop(resolution),
            Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )

    return T.Compose(transforms)


class TrainImageDataset(data.Dataset):
    image_exts = ["JPEG", "jpeg", "jpg", "png"]

    def __init__(
        self,
        image_folder,
        train=True,
        resolution=256,
        cache_file=None,
        is_main_process=False,
        augment=None,
    ):

        self.train = train
        self.resolution = resolution
        self.image_folder = image_folder
        self.cache_file = cache_file
        self.transform = get_transform(self.resolution, augment=augment)
        print("Building datasets...")
        self.is_main_process = is_main_process
        self.samples = self._make_dataset()

    def _make_dataset(self):
        cache_file = osp.join(self.image_folder, self.cache_file)

        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            samples += sum(
                [
                    glob(osp.join(self.image_folder, "**", f"*.{ext}"), recursive=True)
                    for ext in self.image_exts
                ],
                [],
            )
            if self.is_main_process:
                with open(cache_file, "wb") as f:
                    pickle.dump(samples, f)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        # try:
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # C H W
        return dict(image=image, label="")
        # except Exception as e:
        #     print(f"Error with {e}, {image_path}")
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))


class ValidImageDataset(data.Dataset):
    image_exts = ["JPEG", "jpeg", "jpg", "png"]

    def __init__(
        self,
        real_image_dir,
        crop_size_width=None,
        crop_size_height=None,
        crop_size=None,
        resolution=256,
        is_main_process=False,
    ) -> None:
        super().__init__()
        self.is_main_process = is_main_process
        self.real_image_files = self._make_dataset(real_image_dir)

        if crop_size is not None:
            self.crop_size_width = crop_size
            self.crop_size_height = crop_size
        else:
            self.crop_size_width = crop_size_width
            self.crop_size_height = crop_size_height

        self.short_size = resolution
        self.transform = Compose(
            [
                ToTensor(),
                Resize(resolution),
                (
                    CenterCrop((self.crop_size_height, self.crop_size_width))
                    if self.crop_size_width is not None
                    and self.crop_size_height is not None
                    else Lambda(lambda x: x)
                ),
                Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

    def _make_dataset(self, real_image_files):
        cache_file = osp.join(real_image_files, "idx.pkl")

        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            samples += sum(
                [
                    glob(osp.join(real_image_files, "**", f"*.{ext}"), recursive=True)
                    for ext in self.image_exts
                ],
                [],
            )
            if self.is_main_process:
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(samples, f)
                except Exception as e:
                    print(f"Error with {e}, {cache_file}")
        return samples

    def __len__(self):
        return len(self.real_image_files)

    def __getitem__(self, index):
        try:
            if index >= len(self):
                raise IndexError
            real_image_file = self.real_image_files[index]
            real_image_tensor = Image.open(real_image_file).convert("RGB")
            real_image_tensor = self.transform(real_image_tensor)
            image_name = os.path.basename(real_image_file)
            return {"image": real_image_tensor, "file_name": image_name}
        except:
            print(f"Image error: {self.real_image_files[index]}")
            return self.__getitem__(0)
