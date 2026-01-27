import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
import tqdm
from itertools import chain
import wandb
import random
import numpy as np
import shutil
from pathlib import Path
from torch.nn import functional as F
from accelerate.utils import set_seed

from einops import rearrange
from src.model import *
from src.model.ema_model import EMA
from src.dataset.image_dataset import TrainImageDataset, ValidImageDataset
from src.model.utils.module_utils import resolve_str_to_obj
from src.utils.image_utils import tensor01_to_image, combined_image
from src.eval.cal_ssim import calculate_ssim
from src.eval.cal_fid import calculate_frechet_distance, InceptionV3

try:
    import lpips
except:
    raise ModuleNotFoundError("Need lpips to valid.")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def setup_logger(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        f"[rank{rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    return logger


def check_unused_params(model):
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params


def set_requires_grad_optimizer(optimizer, requires_grad):
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad


def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)


def get_exp_name(args):
    return f"{args.exp_name}"


def set_train(modules):
    for module in modules:
        module.train()


def set_eval(modules):
    for module in modules:
        module.eval()


def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)


def compute_grad_norm(model):
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2  # L2范数
    return grad_norm**0.5  # 求平方根得到L2范数


def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
):
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
        },
        filepath,
    )
    return filepath


def all_gather_np(array: np.ndarray):
    """把任意 shape 的 numpy array gather 到 rank，返回 rank拼接后的 ndarray。"""
    local_size = torch.tensor([array.shape[0]], device="cuda")
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(v.item()) for v in size_list]

    # flatten 传输，最后再 reshape
    flat = torch.from_numpy(array).cuda()
    # 先 gather 尺寸最大的 tensor，其他 pad
    max_size = max(sizes) * array.shape[1]
    padding = max_size - flat.numel()
    flat_padded = torch.cat(
        [flat.flatten(), torch.zeros(padding, device="cuda")], dim=0
    )

    gather_list = [torch.empty_like(flat_padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, flat_padded)

    chunks = []
    for g, sz in zip(gather_list, sizes):
        chunks.append(g[: sz * array.shape[1]].view(sz, array.shape[1]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def valid(global_rank, rank, model, val_dataloader, precision, args):
    if args.eval_lpips:
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()
        # if args.compile:
        #     lpips_model = torch.compile(lpips_model)
    if args.eval_fid:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        inception = InceptionV3([block_idx])
        inception.to(rank, dtype=precision)
        # inception = DDP(inception, device_ids=[rank])
        inception.requires_grad_(False)
        inception.eval()
        # if args.compile:
        #     inception = torch.compile(inception)

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    loss_list = []
    psnr_list = []
    lpips_list = []
    ssim_list = []
    feats_ref_list = []
    feats_rec_list = []
    image_log = []
    num_image_log = args.eval_num_image_log

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["image"].to(rank, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=precision):
                output = model(inputs)
                image_recon = output.sample if hasattr(output, "sample") else output[0]

            # Upload images
            if global_rank == 0:
                for i in range(len(image_recon)):
                    if num_image_log <= 0:
                        break
                    image = tensor01_to_image(image_recon[i])
                    image_log.append(image)
                    num_image_log -= 1

            valid_loss = torch.abs(inputs - image_recon).mean()
            loss_list.append(valid_loss.detach().float().cpu().item())

            # Calculate LPIPS, -1, 1
            if args.eval_lpips:
                lpips_score = (
                    lpips_model.forward(inputs, image_recon)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                lpips_list.append(lpips_score)

            if args.eval_psnr:
                inputs = (inputs + 1) / 2
                image_recon = (image_recon + 1) / 2
                # Calculate PSNR, 0-1
                mse = torch.mean(torch.square(inputs - image_recon), dim=(1, 2, 3))
                psnr = 20 * torch.log10(1 / torch.sqrt(mse))
                psnr = psnr.mean().detach().cpu().item()
                psnr_list.append(psnr)

            # Calculate SSIM, 0-1
            if args.eval_ssim:
                ssim = calculate_ssim(
                    inputs.detach().float().cpu().numpy(),
                    image_recon.detach().float().cpu().numpy(),
                    channel_axis=0,
                )
                ssim_list.append(ssim)

            # FID features, 0-1
            if args.eval_fid:
                feats_ref_list.append(
                    inception(inputs.to(dtype=precision))[0]
                    .squeeze(-1)
                    .squeeze(-1)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                feats_rec_list.append(
                    inception(image_recon.to(dtype=precision))[0]
                    .squeeze(-1)
                    .squeeze(-1)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

            if global_rank == 0:
                bar.update()

    return (
        loss_list,
        psnr_list,
        lpips_list,
        ssim_list,
        feats_ref_list,
        feats_rec_list,
        image_log,
    )


def gather_valid_result(
    loss_list,
    psnr_list,
    lpips_list,
    ssim_list,
    feats_ref_list,
    feats_rec_list,
    image_log_list,
    rank,
    world_size,
):
    gathered_loss_list = [None for _ in range(world_size)]
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_ssim_list = [None for _ in range(world_size)]
    gathered_image_logs = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_loss_list, loss_list)
    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_ssim_list, ssim_list)
    gathered_feats_ref_list = all_gather_np(np.concatenate(feats_ref_list, axis=0))
    gathered_feats_rec_list = all_gather_np(np.concatenate(feats_rec_list, axis=0))
    dist.all_gather_object(gathered_image_logs, image_log_list)

    fid_val = 0.0
    if len(feats_ref_list) > 0:
        mu1, mu2 = gathered_feats_ref_list.mean(0), gathered_feats_rec_list.mean(0)
        sigma1 = np.cov(gathered_feats_ref_list, rowvar=False)
        sigma2 = np.cov(gathered_feats_rec_list, rowvar=False)
        fid_val = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return (
        np.array(gathered_loss_list).mean(),
        np.array(gathered_psnr_list).mean(),
        np.array(gathered_lpips_list).mean(),
        np.array(gathered_ssim_list).mean(),
        fid_val,
        list(chain(*gathered_image_logs)),
    )


def train(args):
    # setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger = setup_logger(rank)

    set_random_seed(args.seed)
    # init
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True)
        except:
            logger.warning(f"`{ckpt_dir}` exists!")
    dist.barrier()

    # load generator model
    model_cls = ModelRegistry.get_model(args.model_name)

    if not model_cls:
        raise ModuleNotFoundError(
            f"`{args.model_name}` not in {str(ModelRegistry._models.keys())}."
        )

    if args.pretrained_model_name_or_path is not None:
        if global_rank == 0:
            logger.warning(
                f"You are loading a checkpoint from `{args.pretrained_model_name_or_path}`."
            )
        model = model_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    else:
        if global_rank == 0:
            logger.warning(f"Model will be inited randomly.")
        model = model_cls.from_config(args.model_config)

    if global_rank == 0:
        logger.warning("Connecting to WANDB...")
        model_config = dict(**model.config)
        args_config = dict(**vars(args))
        if "resolution" in model_config:
            del model_config["resolution"]
        print(f"model_config:\n{model_config}")
        print(f"args_config:\n{args_config}")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "t1"),
            config=dict(**model_config, **args_config),
            name=get_exp_name(args),
        )

    dist.barrier()

    # load discriminator model
    disc_cls = resolve_str_to_obj(args.disc_cls, append=False)
    logger.warning(
        f"disc_class: {args.disc_cls} perceptual_weight: {args.perceptual_weight}  loss_type: {args.loss_type}"
    )
    disc = disc_cls(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
    )

    # DDP
    model = model.to(rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if args.compile:
        model = torch.compile(model)
    disc = disc.to(rank)
    disc = DDP(disc, device_ids=[rank], find_unused_parameters=True)
    if args.compile:
        disc = torch.compile(disc)

    # load dataset
    dataset = TrainImageDataset(
        args.image_path,
        resolution=args.resolution,
        cache_file="idx.pkl",  # Cache idx
        is_main_process=global_rank == 0,
        augment=args.augment,
    )

    ddp_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True,
        num_workers=args.dataset_num_worker,
    )

    def get_dataloader(img_eval_path):
        val_dataset = ValidImageDataset(
            real_image_dir=img_eval_path,
            crop_size=args.eval_resolution,
            resolution=args.eval_resolution,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
            drop_last=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=args.dataset_num_worker,
        )
        return val_dataloader

    val_dataloader_dict = {}
    if args.imgnet_eval_path is not None:
        val_dataloader_dict["imgnet"] = get_dataloader(args.imgnet_eval_path)
    if args.coco_eval_path is not None:
        val_dataloader_dict["coco"] = get_dataloader(args.coco_eval_path)

    # optimizer
    modules_to_train = [module for module in model.module.get_decoder()]
    if args.freeze_encoder:
        for module in model.module.get_encoder():
            module.eval()
            module.requires_grad_(False)
        logger.info("Encoder is freezed!")
    else:
        modules_to_train += [module for module in model.module.get_encoder()]

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train += list(
            filter(lambda p: p.requires_grad, module.parameters())
        )

    gen_optimizer = torch.optim.AdamW(
        parameters_to_train,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    # AMP
    precision = torch.bfloat16
    if args.mix_precision == "fp32":
        precision = torch.float32

    # load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(
            checkpoint["state_dict"]["gen_model"], strict=False
        )

        # resume optimizer
        if not args.not_resume_optimizer:
            gen_optimizer.load_state_dict(
                checkpoint["optimizer_state"]["gen_optimizer"]
            )

        # resume discriminator
        if not args.not_resume_discriminator:
            disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
            disc_optimizer.load_state_dict(
                checkpoint["optimizer_state"]["disc_optimizer"]
            )

        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()

    logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0:
        logger.info(f"{model}")
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Encoder:\t{total_params(model.module.encoder):d}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.decoder):d}M")
        logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start training!")

    # training bar
    bar_desc = "Epoch: {current_epoch}, Loss: {loss}"
    bar = None
    if global_rank == 0:
        max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataloader)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    num_epochs = args.epochs

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            bar.update()

    # training Loop
    for epoch in range(num_epochs):
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["image"].to(rank, non_blocking=True)

            # select generator or discrßiminator
            if (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                set_eval(modules_to_train)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                set_train(modules_to_train)
                step_gen = True
                step_dis = False

            assert (
                step_gen or step_dis
            ), "You should backward either Gen. or Dis. in a step."

            # forward
            with torch.amp.autocast("cuda", dtype=precision):
                outputs = model(inputs)
                recon = outputs.sample
                posterior = outputs.latent_dist  # when fsq, it is None
                lfq_extra_output = outputs.lfq_extra_output  # when lfq
                vq_extra_output = outputs.vq_extra_output  # when vq

            # generator loss
            if step_gen:
                with torch.amp.autocast("cuda", dtype=precision):
                    g_loss, g_log = disc(
                        inputs=inputs,
                        reconstructions=recon,
                        posteriors=posterior,  # None means kl_loss=0
                        loss_break=lfq_extra_output or vq_extra_output,  # when lfq
                        optimizer_idx=0,  # 0 - generator
                        global_step=current_step,
                        last_layer=model.module.get_last_layer(),
                        split="train",
                    )
                gen_optimizer.zero_grad()
                g_loss.backward()
                if args.max_norm is not None and args.max_norm > 0:
                    clip_grad_norm_(model.parameters(), args.max_norm)
                grad_norm = compute_grad_norm(model)
                gen_optimizer.step()

                # update ema
                if args.ema:
                    ema.update()

            # log to wandb
            if global_rank == 0 and current_step % args.log_steps == 0:
                log_data = {
                    "train/generator_loss": float(g_loss.item()),
                    "train/rec_loss": float(g_log["train/rec_loss"]),
                    "train/lpips_loss": float(g_log["train/lpips_loss"]),
                    "train/grad_norm": float(grad_norm),
                }
                if "train/codebook_loss" in g_log:
                    log_data["train/codebook_loss"] = g_log["train/codebook_loss"]
                if "train/per_sample_entropy" in g_log:
                    log_data["train/per_sample_entropy"] = g_log[
                        "train/per_sample_entropy"
                    ]
                if "train/codebook_entropy" in g_log:
                    log_data["train/codebook_entropy"] = g_log["train/codebook_entropy"]
                if "train/commit_loss" in g_log:
                    log_data["train/commit_loss"] = g_log["train/commit_loss"]
                if posterior is not None:
                    latent_stats = posterior.sample()
                    mean_list = (
                        torch.mean(latent_stats, dim=[0, 2, 3]).flatten().tolist()
                    )
                    std_list = torch.std(latent_stats, dim=[0, 2, 3]).flatten().tolist()
                else:
                    mean_list = (
                        model.module.quantize.running_mean.detach()
                        .float()
                        .flatten()
                        .tolist()
                    )
                    var_list = (
                        model.module.quantize.running_var.detach()
                        .float()
                        .flatten()
                        .tolist()
                    )
                    std_list = [i**0.5 for i in var_list]

                for d, m in enumerate(mean_list):
                    log_data[f"latent_mean/dim-{d}"] = float(m)
                for d, v in enumerate(std_list):
                    log_data[f"latent_std/dim-{d}"] = float(v)
                wandb.log(log_data, step=current_step)

            # discriminator loss
            if step_dis:
                with torch.amp.autocast("cuda", dtype=precision):
                    d_loss, d_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=1,
                        global_step=current_step,
                        last_layer=None,
                        split="train",
                    )
                disc_optimizer.zero_grad()
                d_loss.backward()
                disc_optimizer.step()

                if global_rank == 0 and current_step % args.log_steps == 0:
                    wandb.log(
                        {"train/discriminator_loss": d_loss.item()}, step=current_step
                    )

            update_bar(bar)
            current_step += 1

            # valid model

            def valid_model(model, val_dataloader, name=""):
                set_modules_requires_grad(modules_to_train, False)
                set_eval(modules_to_train)
                (
                    loss_list,
                    psnr_list,
                    lpips_list,
                    ssim_list,
                    feats_ref_list,
                    feats_rec_list,
                    image_log,
                ) = valid(global_rank, rank, model, val_dataloader, precision, args)
                (
                    valid_loss,
                    valid_psnr,
                    valid_lpips,
                    valid_ssim,
                    valid_fid,
                    valid_image_log,
                ) = gather_valid_result(
                    loss_list,
                    psnr_list,
                    lpips_list,
                    ssim_list,
                    feats_ref_list,
                    feats_rec_list,
                    image_log,
                    rank,
                    dist.get_world_size(),
                )
                if global_rank == 0:
                    name = "_" + name if name != "" else name
                    wandb.log(
                        {
                            f"val{name}/recon": wandb.Image(
                                combined_image(np.array(valid_image_log)).transpose(
                                    1, 2, 0
                                )
                            )
                        },
                        step=current_step,
                    )
                    wandb.log({f"val{name}/loss": valid_loss}, step=current_step)
                    if args.eval_psnr:
                        wandb.log({f"val{name}/psnr": valid_psnr}, step=current_step)
                    if args.eval_lpips:
                        wandb.log({f"val{name}/lpips": valid_lpips}, step=current_step)
                    if args.eval_ssim:
                        wandb.log({f"val{name}/ssim": valid_ssim}, step=current_step)
                    if args.eval_fid:
                        wandb.log({f"val{name}/fid": valid_fid}, step=current_step)
                    logger.info(f"{name} Validation done.")

            if current_step % args.eval_steps == 0:
                if global_rank == 0:
                    logger.info("Starting validation...")
                for dataset_name, val_dataloader in val_dataloader_dict.items():
                    valid_model(
                        model, val_dataloader, "coco" if dataset_name == "coco" else ""
                    )
                    if args.ema:
                        ema.apply_shadow()
                        valid_model(
                            model,
                            val_dataloader,
                            "ema_coco" if dataset_name == "coco" else "ema",
                        )
                        ema.restore()

            # save checkpoint
            if current_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "dics_model": disc.module.state_dict(),
                    },
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.shadow if args.ema else {},
                )
                shutil.copyfile(
                    args.model_config, os.path.join(ckpt_dir, "config.json")
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")

    # end training
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    # Exp setting
    parser.add_argument(
        "--exp_name", type=str, default="test", help="number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="number of epochs to train"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="")
    parser.add_argument("--ckpt_dir", type=str, default="./results", help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="weight decay")
    parser.add_argument("--beta2", type=float, default=0.999, help="weight decay")
    parser.add_argument("--log_steps", type=int, default=5, help="log steps")
    parser.add_argument("--freeze_encoder", action="store_true", help="")
    parser.add_argument("--max_norm", type=float, default=1.0, help="")

    # Data
    parser.add_argument("--augment", type=str, default=None, help="")
    parser.add_argument("--image_path", type=str, default=None, help="")
    parser.add_argument("--imgnet_eval_path", type=str, default=None, help="")
    parser.add_argument("--coco_eval_path", type=str, default=None, help="")
    parser.add_argument("--resolution", type=int, default=256, help="")
    # Generator model
    parser.add_argument("--ignore_mismatched_sizes", action="store_true", help="")
    parser.add_argument("--find_unused_parameters", action="store_true", help="")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help=""
    )
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument("--not_resume_training_process", action="store_true", help="")
    parser.add_argument("--model_config", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--not_resume_discriminator", action="store_true", help="")
    parser.add_argument("--not_resume_optimizer", action="store_true", help="")
    # Discriminator Model
    parser.add_argument("--load_disc_from_checkpoint", type=str, default=None, help="")
    parser.add_argument(
        "--disc_cls",
        type=str,
        default="src.model.losses.LPIPSWithDiscriminator2D",
        help="",
    )
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")
    parser.add_argument("--loss_type", type=str, default="l1", help="")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="")
    parser.add_argument("--eval_num_image_log", type=int, default=4, help="")
    parser.add_argument("--eval_psnr", action="store_true", help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")
    parser.add_argument("--eval_ssim", action="store_true", help="")
    parser.add_argument("--eval_fid", action="store_true", help="")

    parser.add_argument("--compile", action="store_true", help="")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=8, help="")

    # EMA
    parser.add_argument("--ema", action="store_true", help="")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
