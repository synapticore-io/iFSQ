import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import json
from copy import deepcopy
from src.model import *
from src.dataset.image_dataset import TrainImageDataset, ValidImageDataset
from train_ddp import (
    valid,
    gather_valid_result,
    setup_logger,
    ddp_setup,
    total_params,
    set_random_seed,
)


def evaluate(args):
    # setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger = setup_logger(rank)

    # init
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # load generator model
    model_cls = ModelRegistry.get_model(args.model_name)

    if not model_cls:
        raise ModuleNotFoundError(
            f"`{args.model_name}` not in {str(ModelRegistry._models.keys())}."
        )

    print(f"You are loading a checkpoint from `{args.ckpt_path}`.")
    model = model_cls.from_config(args.model_config)
    model.load_state_dict(
        {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in ckpt["state_dict"]["gen_model"].items()
        }
    )

    if args.ema:
        ema_model = deepcopy(model)
        ema_model.load_state_dict(
            {
                k.replace("_orig_mod.", "").replace("module.", ""): v
                for k, v in ckpt["ema_state_dict"].items()
            },
            strict=False,
        )

    # DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    if args.compile:
        model = torch.compile(model)
    if args.ema:
        ema_model = ema_model.to(rank)
        ema_model = DDP(ema_model, device_ids=[rank])
        if args.compile:
            ema_model = torch.compile(ema_model)

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

    # AMP
    precision = torch.bfloat16
    if args.mix_precision == "fp32":
        precision = torch.float32

    logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0:
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Encoder:\t{total_params(model.module.encoder):d}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.decoder):d}M")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start eval!")

    @torch.inference_mode()
    def valid_model(model, val_dataloader, name=""):
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
            logger.info(f"{name} Validation done.")
        return dict(
            valid_loss=float(valid_loss),
            valid_psnr=float(valid_psnr),
            valid_lpips=float(valid_lpips),
            valid_ssim=float(valid_ssim),
            valid_fid=float(valid_fid),
        )

    if global_rank == 0:
        logger.info("Starting validation...")

    res_dicts = {}
    for dataset_name, val_dataloader in val_dataloader_dict.items():
        res_dicts[dataset_name] = {}
        with torch.amp.autocast("cuda", dtype=precision):
            res_dict = valid_model(model, val_dataloader)
            res_dicts[dataset_name]["model"] = res_dict
            if args.ema:
                res_dict = valid_model(ema_model, val_dataloader, "ema")
                res_dicts[dataset_name]["ema_model"] = res_dict

    with open(ckpt_path.replace(".ckpt", ".json"), "w") as f:
        json.dump(res_dicts, f, indent=1)
    # end training
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    # Exp setting
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument("--ckpt_path", type=str, default="./results/1.ckpt", help="")

    # Data
    parser.add_argument("--imgnet_eval_path", type=str, default=None, help="")
    parser.add_argument("--coco_eval_path", type=str, default=None, help="")
    parser.add_argument("--resolution", type=int, default=256, help="")
    # Generator model
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--model_config", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="precision for training",
    )

    # Validation
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="")
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

    args = parser.parse_args()

    set_random_seed(args.seed)
    evaluate(args)


if __name__ == "__main__":
    main()
