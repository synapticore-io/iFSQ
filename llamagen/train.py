# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import random
import math
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faste
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from accelerate.utils import ProjectConfiguration
import yaml
import numpy as np
import logging
import os
import argparse
from time import time
from datetime import datetime
from glob import glob
from einops import rearrange
from copy import deepcopy
from PIL import Image
from collections import OrderedDict

# local imports
from models import Models, VQ_models, FSQ_models, CusVQ_models
from accelerate import Accelerator
from datasets import ImgLatentDataset, ImgDataset
from tools.repa import load_encoders


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def load_weights_with_shape_check(model, checkpoint, rank=0):
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint["model"].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                if rank == 0:
                    print(
                        f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}"
                    )
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)

    return model


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


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def get_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def do_train(train_config):
    """
    Trains a DiT.
    """
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    precision = torch.bfloat16
    if train_config["train"]["precision"] == "fp32":
        precision = torch.float32

    set_seed(train_config["train"]["seed"], global_rank, device_specific=True)
    # Setup an experiment folder:
    # if global_rank == 0:
    #     logger = create_logger(train_config["train"]["output_dir"])

    checkpoint_dir = f"{train_config['train']['output_dir']}/checkpoints"
    os.makedirs(
        checkpoint_dir, exist_ok=True
    )  # Make results folder (holds all experiment subfolders)

    offline_features = (
        train_config["data"]["offline_features"]
        if "offline_features" in train_config["data"]
        else False
    )
    # Create model:
    # if not offline_features:

    fsq_model = None
    vq_model = None
    factorized_n_layer = 2
    if "fsq" in train_config:
        fsq_model = FSQ_models["fsq"](
            train_config["fsq"]["config_path"],
            train_config["fsq"]["factorized_bits"],
        )
        factorized_n_layer = (
            train_config["fsq"]["factorized_n_layer"]
            if "factorized_n_layer" in train_config["fsq"]
            else 2
        )
        checkpoint = torch.load(train_config["fsq"]["model_path"], map_location="cpu")[
            "ema_state_dict"
        ]
        checkpoint = {k.replace("module.", "model."): v for k, v in checkpoint.items()}
        msg = fsq_model.load_state_dict(checkpoint, strict=False)
        print(msg)
        fsq_model.to(device)
        fsq_model.eval()

        loss_weights = (
            train_config["fsq"]["loss_weights"]
            if "loss_weights" in train_config["fsq"]
            else None
        )
    elif "cusvq" in train_config:
        vq_model = CusVQ_models["cusvq"](
            train_config["cusvq"]["config_path"],
        )
        checkpoint = torch.load(
            train_config["cusvq"]["model_path"], map_location="cpu"
        )["ema_state_dict"]
        checkpoint = {k.replace("module.", "model."): v for k, v in checkpoint.items()}
        msg = vq_model.load_state_dict(checkpoint, strict=False)
        print(msg)
        vq_model.to(device)
        vq_model.eval()
    else:
        vq_model = VQ_models[train_config["vqvae"]["vq_model"]](
            codebook_size=train_config["vqvae"]["codebook_size"],
            codebook_embed_dim=train_config["vqvae"]["codebook_embed_dim"],
        )
        checkpoint = torch.load(train_config["vqvae"]["model_path"], map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        vq_model.to(device)
        vq_model.eval()

    if "cusvq" in train_config:
        downsample_ratio = train_config["cusvq"]["downsample_ratio"]
        vocab_size = train_config["cusvq"]["codebook_size"]
    elif "vqvae" in train_config and "downsample_ratio" in train_config["vqvae"]:
        downsample_ratio = train_config["vqvae"]["downsample_ratio"]
        vocab_size = train_config["vqvae"]["codebook_size"]
    elif "fsq" in train_config and "downsample_ratio" in train_config["fsq"]:
        downsample_ratio = train_config["fsq"]["downsample_ratio"]
        vocab_size = fsq_model.vocab_size()
    else:
        downsample_ratio = 8
    assert (
        train_config["data"]["image_size"] % downsample_ratio == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config["data"]["image_size"] // downsample_ratio

    print("vocab_size", vocab_size)

    use_repa = (
        train_config["repa"]["use_repa"]
        if "use_repa" in train_config["repa"]
        else False
    )
    proj_coef = 0.5
    if use_repa:
        proj_coef = (
            train_config["repa"]["proj_coef"]
            if "proj_coef" in train_config["repa"]
            else 0.5
        )
    eval_cknna = (
        train_config["sample"]["eval_cknna"]
        if "eval_cknna" in train_config["sample"]
        else False
    )
    if eval_cknna or (use_repa and train_config["repa"]["enc_type"] != None):
        encoders, encoder_types, architectures = load_encoders(
            train_config["repa"]["enc_type"],
            device,
            train_config["data"]["image_size"],
            downsample_ratio,
        )
    z_dims = [encoder.embed_dim for encoder in encoders] if use_repa else None
    encoder_depth = train_config["repa"]["encoder_depth"] if use_repa else None

    if train_config["model"]["drop_path_rate"] > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = train_config["model"]["dropout_p"]
    kwargs = dict(
        block_size=latent_size**2,
        vocab_size=vocab_size,
        num_classes=train_config["data"]["num_classes"],
        cls_token_num=(
            train_config["model"]["cls_token_num"]
            if "cls_token_num" in train_config["model"]
            else 1
        ),
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=(
            train_config["model"]["drop_path_rate"]
            if "drop_path_rate" in train_config["model"]
            else 0.0
        ),
        token_dropout_p=(
            train_config["model"]["token_dropout_p"]
            if "token_dropout_p" in train_config["model"]
            else 0.1
        ),
        use_checkpoint=(
            train_config["model"]["use_checkpoint"]
            if "use_checkpoint" in train_config["model"]
            else False
        ),
        z_dims=z_dims,
        encoder_depth=encoder_depth,
        factorized_n_layer=factorized_n_layer,
    )
    model = Models[train_config["model"]["model_type"]](**kwargs)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 打印当前显存占用
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # MB

    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Reserved Memory: {reserved:.2f} MB")
    if train_config["train"]["wandb"] and global_rank == 0:
        import wandb

        wandb.login(key=train_config["wandb"]["key"])
        wandb.init(
            project=train_config["wandb"]["proj_name"],
            name=train_config["wandb"]["log_name"],
            config=train_config,
        )

    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    # load pretrained model
    if "weight_init" in train_config["train"]:
        checkpoint = torch.load(
            train_config["train"]["weight_init"],
            map_location=lambda storage, loc: storage,
        )
        # remove the prefix 'module.' from the keys
        checkpoint["model"] = {
            k.replace("module.", ""): v for k, v in checkpoint["model"].items()
        }
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if global_rank == 0:
            print(
                f"Loaded pretrained model from {train_config['train']['weight_init']}"
            )

    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[device])

    if global_rank == 0:
        print(f"Model: {model}")
        print(
            f"LightingLlamaGen Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
        )
        print(
            f"LightingLlamaGen Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print(
            f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}"
        )
        for name, param in model.named_parameters():
            print(f"{name+'.requires_grad':<60}: {param.requires_grad}")

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config["optimizer"]["lr"],
        weight_decay=train_config["optimizer"]["weight_decay"],
        betas=(train_config["optimizer"]["beta1"], train_config["optimizer"]["beta2"]),
    )

    # Setup data
    if offline_features:
        dataset = ImgLatentDataset(data_dir=train_config["data"]["data_path"])
    else:
        crop_size = train_config["data"]["image_size"]
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: random_crop_arr(pil_image, crop_size)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        dino_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: (x + 1) / 2),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                transforms.Lambda(
                    lambda x: F.interpolate(
                        x.unsqueeze(0),
                        224 * (x.shape[-1] // 256) * (16 // downsample_ratio),
                        mode="bicubic",
                    ).squeeze(0)
                ),
            ]
        )
        dataset = ImgDataset(
            train_config["data"]["data_path"],
            transform=transform,
            dino_transform=dino_transform,
        )

    ddp_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True
    )
    assert train_config["train"]["global_batch_size"] % world_size == 0
    batch_size_per_gpu = train_config["train"]["global_batch_size"] // world_size
    global_batch_size = batch_size_per_gpu * world_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        num_workers=train_config["data"]["num_workers"],
        pin_memory=True,
        sampler=ddp_sampler,
        drop_last=True,
    )
    if global_rank == 0:
        print(
            f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}"
        )
        print(
            f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size"
        )

    # Prepare models for training:
    update_ema(
        ema, model.module, decay=0
    )  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    train_config["train"]["resume"] = (
        train_config["train"]["resume"] if "resume" in train_config["train"] else False
    )

    if train_config["train"]["resume"]:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(
                latest_checkpoint, map_location=lambda storage, loc: storage
            )
            model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["opt"])
            ema.load_state_dict(checkpoint["ema"])
            train_steps = int(latest_checkpoint.split("/")[-1].split(".")[0])
            if global_rank == 0:
                print(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if global_rank == 0:
                print("No checkpoint found. Starting training from scratch.")

    # Variables for monitoring/logging purposes:
    if not train_config["train"]["resume"]:
        train_steps = 0
    log_steps = 0
    eval_every = (
        train_config["train"]["eval_every"]
        if "eval_every" in train_config["train"]
        else math.inf
    )
    start_time = time()
    if global_rank == 0:
        print(f"Train config: {train_config}")

    for epoch in range(train_config["train"]["epochs"]):
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        for x, y, dino_x in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if offline_features:
                z_indices = x
            else:
                with torch.amp.autocast("cuda", dtype=precision):
                    with torch.no_grad():
                        if vq_model is not None:
                            _, _, [_, _, indices] = vq_model.encode(x)
                            z_indices = indices.reshape(x.shape[0], -1)
                        else:
                            _, _, indices = fsq_model.encode(x)
                            # b h w g
                            z_indices = rearrange(indices, "b h w g -> g b (h w)")
                            z_indices = [i for i in z_indices]

                        if use_repa:
                            dino_x = dino_x.to(device, non_blocking=True)
                            zs = []
                            for encoder, encoder_type, arch in zip(
                                encoders, encoder_types, architectures
                            ):
                                z = encoder.forward_features(dino_x)
                                assert "dinov2" in encoder_type
                                z = z["x_norm_patchtokens"]
                                zs.append(z)

            # print(z_indices.shape, "z_indices")
            c_indices = y.reshape(-1)
            if (
                isinstance(z_indices, list)
                and len(z_indices) == 1
                and z_indices[0].ndim == 2
            ):
                z_indices = z_indices[0]
                assert z_indices.shape[0] == c_indices.shape[0]
            elif isinstance(z_indices, list):
                assert z_indices[0].shape[0] == c_indices.shape[0]
            else:
                assert z_indices.shape[0] == c_indices.shape[0]

            with torch.amp.autocast("cuda", dtype=precision):
                logits, zs_tilde = model(cond_idx=c_indices, idx=z_indices)

            # projection loss
            proj_loss = 0.0
            if use_repa:
                bsz = zs[0].shape[0]
                for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                    for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                        z_tilde_j = F.normalize(z_tilde_j, dim=-1)
                        z_j = F.normalize(z_j, dim=-1)
                        # print(
                        #     f"z_j.shape: {z_j.shape}, z_tilde_j.shape: {z_tilde_j.shape}"
                        # )
                        proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                proj_loss /= len(zs) * bsz

            ar_losses = []
            if isinstance(logits, list):
                ar_loss = 0
                for i in range(len(z_indices)):
                    logits_i, target_i = logits[i], z_indices[i]
                    loss_i = F.cross_entropy(
                        logits_i.reshape(-1, logits_i.size(-1)), target_i.reshape(-1)
                    )
                    ar_loss += loss_i * (
                        loss_weights[i] if loss_weights is not None else 1.0
                    )
                    ar_losses.append(loss_i)
            else:
                ar_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), z_indices.view(-1)
                )
            loss = ar_loss + proj_loss * proj_coef

            opt.zero_grad()
            loss.backward()
            if "max_grad_norm" in train_config["optimizer"]:
                clip_grad_norm_(
                    model.parameters(), train_config["optimizer"]["max_grad_norm"]
                )

            opt.step()
            update_ema(ema, model.module)

            log_steps += 1
            train_steps += 1
            if train_steps % train_config["train"]["log_every"] == 0:
                avg_loss = ar_loss.item()
                avg_proj_loss = proj_loss.item() if use_repa else 0.0
                avg_grad_norm = get_grad_norm(model)
                if global_rank == 0:
                    print(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: (step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Proj Loss: {avg_proj_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}"
                    )
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "proj_loss": avg_proj_loss,
                            "grad_norm": avg_grad_norm,
                        },
                        step=train_steps,
                    )
                    if len(ar_losses) > 0:
                        for g_i, loss_group_i in enumerate(ar_losses):
                            wandb.log({f"loss_g{g_i}": loss_group_i}, step=train_steps)
                # Reset monitoring variables:
                log_steps = 0

            # Save checkpoint:
            if (
                train_steps % train_config["train"]["ckpt_every"] == 0
                and train_steps > 0
            ):
                if global_rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if global_rank == 0:
                        print(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            # Eval online or last step:
            if train_steps % eval_every == 0 and train_steps > 0:
                from inference import do_sample

                # stored
                temp_stored_params = [
                    param.detach().cpu().clone() for param in model.parameters()
                ]
                # copy ema to model
                for s_param, param in zip(ema.parameters(), model.parameters()):
                    param.data.copy_(s_param.to(param.device).data)
                # sampling
                model.eval()
                temp_cfg = train_config["sample"]["cfg_scale"]
                train_config["sample"]["cfg_scale"] = 1.0
                with torch.no_grad():
                    sample_folder_dir, cknna_scores = do_sample(
                        train_config,
                        global_rank,
                        world_size,
                        ckpt_path=f"{train_steps:07d}.pt",
                        model=model.module,
                        vq_model=vq_model,
                        fsq_model=fsq_model,
                        downsample_ratio=downsample_ratio,
                        dino_model=encoders[0] if eval_cknna else None,
                        eval_cknna=eval_cknna,
                    )
                print("sample_folder_dir", sample_folder_dir)
                train_config["sample"]["cfg_scale"] = temp_cfg
                model.train()
                # restored
                for c_param, param in zip(temp_stored_params, model.parameters()):
                    param.data.copy_(c_param.data)
                temp_stored_params = None

                # calculate FID
                # Important: FID is only for reference, please use ADM evaluation for paper reporting
                if global_rank == 0:
                    from tools.calculate_fid import calculate_fid_given_paths

                    print(
                        f"Calculating FID with {train_config['sample']['fid_num']} number of samples"
                    )
                    assert (
                        "fid_reference_file" in train_config["data"]
                    ), "fid_reference_file must be specified in config"
                    fid_reference_file = train_config["data"]["fid_reference_file"]
                    fid = calculate_fid_given_paths(
                        [fid_reference_file, sample_folder_dir],
                        batch_size=200,
                        dims=2048,
                        device="cuda",
                        num_workers=16,
                        sp_len=train_config["sample"]["fid_num"],
                    )
                    print(f"(step={train_steps:07d}), Fid={fid}")
                    wandb.log({"fid": fid}, step=train_steps)
                    if eval_cknna:
                        for idx, cknna_score in enumerate(cknna_scores):
                            wandb.log(
                                {f"cknna/layer-{idx*2}": cknna_score}, step=train_steps
                            )
            dist.barrier()
    if global_rank == 0:
        print("Done!")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def set_seed(seed, rank, device_specific=True):
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    args = parser.parse_args()

    train_config = load_config(args.config)

    do_train(train_config)
