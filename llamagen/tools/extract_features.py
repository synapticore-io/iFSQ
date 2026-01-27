# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os
from einops import rearrange
import math
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
from models import VQ_models, FSQ_models


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    ddp_setup()
    # Setup DDP:
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(
            os.path.join(args.code_path, f"{args.dataset}{args.image_size}_codes"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(args.code_path, f"{args.dataset}{args.image_size}_labels"),
            exist_ok=True,
        )

    # create and load model

    vq_model, fsq_model = None, None
    if args.fsq_model is not None:
        fsq_model = FSQ_models["fsq"](
            args.fsq_model,
            args.factorized_bits,
        )
        checkpoint = torch.load(args.fsq_ckpt, map_location="cpu")["ema_state_dict"]
        checkpoint = {k.replace("module.", "model."): v for k, v in checkpoint.items()}
        msg = fsq_model.load_state_dict(checkpoint, strict=False)
        print(msg)
        fsq_model.to(device)
        fsq_model.eval()
    else:
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
        )
        vq_model.to(device)
        vq_model.eval()
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Setup data:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, crop_size)
                ),
                transforms.TenCrop(args.image_size),  # this is a tuple of PIL Images
                transforms.Lambda(
                    lambda crops: torch.stack(
                        [transforms.ToTensor()(crop) for crop in crops]
                    )
                ),  # returns a 4D tensor
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
    else:
        crop_size = args.image_size
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, crop_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,  # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total = 0
    for x, y in loader:
        x = x.to(device)
        if args.ten_crop:
            x_all = x.flatten(0, 1)
            num_aug = 10
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            num_aug = 2
        y = y.to(device)
        with torch.no_grad():
            if fsq_model is not None:
                _, _, indices = fsq_model.encode(x_all)
                codes = rearrange(indices, "(b n) h w g -> b n (h w) g", b=x.shape[0])
            else:
                _, _, [_, _, indices] = vq_model.encode(x_all)
                codes = indices.reshape(x.shape[0], num_aug, -1)

        x = (
            codes.detach().cpu().numpy()
        )  # (1, num_aug, args.image_size//16 * args.image_size//16)
        train_steps = rank + total
        np.save(
            f"{args.code_path}/{args.dataset}{args.image_size}_codes/{train_steps}.npy",
            x,
        )

        y = y.detach().cpu().numpy()  # (1,)
        np.save(
            f"{args.code_path}/{args.dataset}{args.image_size}_labels/{train_steps}.npy",
            y,
        )
        total += dist.get_world_size()
        print(total)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument(
        "--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16"
    )
    parser.add_argument(
        "--vq-ckpt",
        type=str,
        required=False,
        default=None,
        help="ckpt path for vq model",
    )
    parser.add_argument("--fsq-model", type=str, default=None)
    parser.add_argument(
        "--fsq-ckpt",
        type=str,
        required=False,
        default=None,
        help="ckpt path for vq model",
    )
    parser.add_argument(
        "--factorized-bits",
        type=int,
        nargs="*",
        default=None,
        help="输入多个整数作为列表",
    )

    parser.add_argument(
        "--codebook-size",
        type=int,
        default=16384,
        help="codebook size for vector quantization",
    )
    parser.add_argument(
        "--codebook-embed-dim",
        type=int,
        default=8,
        help="codebook dimension for vector quantization",
    )
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument(
        "--image-size", type=int, choices=[256, 384, 448, 512], default=256
    )
    parser.add_argument(
        "--ten-crop", action="store_true", help="whether using random crop"
    )
    parser.add_argument(
        "--crop-range", type=float, default=1.1, help="expanding range of center crop"
    )
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()
    main(args)


"""
extract_codes_c2i.py \
    --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
    --data-path /path/to/imagenet/train \
    --code-path /path/to/imagenet_code_c2i_flip_ten_crop \
    --ten-crop \
    --crop-range 1.1 \
    --image-size 256
extract_codes_c2i.py \
    --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
    --data-path /path/to/imagenet/train \
    --code-path /path/to/imagenet_code_c2i_flip_ten_crop_105 \
    --ten-crop \
    --crop-range 1.05 \
    --image-size 256
"""
