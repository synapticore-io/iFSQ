# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import transforms

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)
from ifsq.src.model.vae.modeling_imagevae import ImageVAEModel


class Cus_AE(nn.Module):
    def __init__(self, ckpt_path, config_path):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        print(self.config)
        self.model = ImageVAEModel.from_config(self.config)
        ckpt = torch.load(ckpt_path)["ema_state_dict"]
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        msg = self.model.load_state_dict(ckpt)
        print(msg)

    def img_transform(self, p_hflip=0, img_size=None):
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
        return transforms.Compose(img_transforms)

    @torch.compile
    def encode_images(self, images):
        with torch.no_grad():
            return self.model.encode(images.cuda()).latent_dist.sample()

    def decode_to_images(self, z):
        with torch.no_grad():
            images = self.model.decode(z.cuda()).sample
            images = (
                torch.clamp(127.5 * images + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )
        return images


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )
