# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from safetensors import safe_open
from torchvision.datasets import ImageFolder


class ImgLatentDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.feature_dir = feature_dir = os.path.join(data_dir, "imagenet256_codes")
        self.label_dir = label_dir = os.path.join(data_dir, "imagenet256_labels")
        self.flip = "flip" in self.feature_dir

        aug_feature_dir = feature_dir.replace("ten_crop/", "ten_crop_105/")
        aug_label_dir = label_dir.replace("ten_crop/", "ten_crop_105/")
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]
        # self.feature_files = [f"{i}.npy" for i in range(50000)] * 100
        # self.label_files = [f"{i}.npy" for i in range(50000)] * 100

    def __len__(self):
        assert len(self.feature_files) == len(
            self.label_files
        ), "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir

        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features).squeeze(0), torch.from_numpy(labels)


class ImgDataset(ImageFolder):
    def __init__(
        self,
        root=None,
        transform=None,
        dino_transform=None,
    ):
        super().__init__(root=root, transform=transform)
        self.dino_transform = dino_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        dino_sample = self.dino_transform(sample)
        return sample, target, dino_sample


if __name__ == "__main__":
    from torchvision.utils import save_image

    data_dir = r"datasets/offline_feature/offline_vqvae16_256_2x10crop_val/imagenet_train_256"
    ckpt_path = r"checkpoints/vq_ds16_c2i.pt"
    dataset = ImgLatentDataset(data_dir)
    num_sample_to_vis = 4
    samples = torch.stack([dataset.__getitem__(i)[0] for i in range(num_sample_to_vis)])
    # from models import VQ_models
    # model = VQ_models["VQ-16"]()
    # ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    # model.load_state_dict(ckpt)
    # print(samples, samples.shape)
    # shape = (4, 8, 16, 16)
    # samples = model.decode_code(samples, shape)
    # print(samples.shape, samples.max(), samples.min(), samples.mean(), samples.std())
    # save_image(samples, "check.png", nrow=4, normalize=True, value_range=(-1, 1))
