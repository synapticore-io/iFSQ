# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import os, math, argparse, yaml, torch, numpy as np
from time import strftime
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from torch.nn import functional as F
from torchvision import transforms

# local imports
from models import FSQ_models
from models import Models, VQ_models, generate, generate_group
from tools.save_npz import create_npz_from_sample_folder
from tools.cknna import cknna
from train import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)
dino_transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: (torch.clamp(x, -1, 1) + 1) / 2),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        transforms.Lambda(
            lambda x: F.interpolate(x, 224 * (x.shape[-1] // 256), mode="bicubic")
        ),
    ]
)


# sample function
def do_sample(
    train_config,
    global_rank,
    world_size,
    ckpt_path=None,
    cfg_scale=None,
    model=None,
    fsq_model=None,
    vq_model=None,
    downsample_ratio=8,
    demo_sample_mode=False,
    dino_model=None,
    eval_cknna=False,
):
    """
    Run sampling.
    """
    device = torch.device(f"cuda:{global_rank % torch.cuda.device_count()}")
    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}".lower()

    if cfg_scale is None:
        cfg_scale = train_config["sample"]["cfg_scale"]
    cfg_interval = (
        train_config["sample"]["cfg_interval"]
        if "cfg_interval" in train_config["sample"]
        else -1
    )
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval:.2f}" + f"-cfg{cfg_scale:.2f}"

    top_k = train_config["sample"]["top_k"]
    top_p = train_config["sample"]["top_p"]
    temperature = train_config["sample"]["temperature"]
    folder_name += f"-topk-{top_k}-topp-{top_p}-temp-{temperature}"

    image_size_eval = train_config["sample"]["image_size"]
    image_size = train_config["data"]["image_size"]

    if demo_sample_mode:
        cfg_interval = -1
        cfg_scale = 4.0
        top_k = 2000
        top_p = 1.0
        temperature = 1.0

    output_dir = train_config["train"]["output_dir"]
    sample_folder_dir = os.path.join(output_dir, folder_name)
    if global_rank == 0:
        if not demo_sample_mode:
            print_with_prefix("Sample_folder_dir=", sample_folder_dir)
        print_with_prefix("ckpt_path=", ckpt_path)
        print_with_prefix("cfg_scale=", cfg_scale)
        print_with_prefix("cfg_interval=", cfg_interval)

    if not os.path.exists(sample_folder_dir):
        if global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            if not demo_sample_mode:
                os.makedirs(sample_folder_dir, exist_ok=True)
    # else:
    #     png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith(".png")]
    #     png_count = len(png_files)
    #     if png_count > train_config["sample"]["fid_num"] and not demo_sample_mode:
    #         if global_rank == 0:
    #             print_with_prefix(
    #                 f"Found {png_count} PNG files in {sample_folder_dir}, skip sampling."
    #             )
    #         return sample_folder_dir, []

    torch.backends.cuda.matmul.allow_tf32 = (
        True  # True: fast but may lead to some small numerical differences
    )
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    seed = train_config["train"]["global_seed"]
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(
        f"Starting rank={global_rank}, seed={seed}, world_size={world_size}."
    )

    # Load model:
    latent_size = train_config["data"]["image_size"] // downsample_ratio

    if global_rank == 0:
        print_with_prefix("Loaded VQVAE model")

    using_cfg = cfg_scale > 1.0
    if using_cfg:
        if global_rank == 0:
            print_with_prefix("Using cfg:", using_cfg)

    if global_rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if not demo_sample_mode:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = train_config["sample"]["per_proc_batch_size"]
    global_batch_size = n * world_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = len(
        [
            name
            for name in os.listdir(sample_folder_dir)
            if (
                os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name
            )
        ]
    )
    total_samples = int(
        math.ceil(train_config["sample"]["fid_num"] / global_batch_size)
        * global_batch_size
    )
    if not demo_sample_mode and global_rank == 0:
        print_with_prefix(
            f"Total number of images that will be sampled: {total_samples}"
        )
    assert (
        total_samples % world_size == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // world_size)
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int(int(num_samples // world_size) // n)
    pbar = range(iterations)
    if not demo_sample_mode:
        pbar = tqdm(pbar) if global_rank == 0 else pbar
    total = 0

    if demo_sample_mode:
        if global_rank == 0:
            images = []
            for label in tqdm(
                [207, 360, 387, 974, 88, 979, 417, 279], desc="Generating Demo Samples"
            ):
                c_indices = torch.tensor([label], device=device)
                gen_fun = generate_group if model.token_factorization else generate
                index_sample = gen_fun(
                    model,
                    c_indices,
                    latent_size**2,
                    cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_logits=True,
                )
                if vq_model is not None:
                    qzshape = [
                        len(c_indices),
                        -1,
                        latent_size,
                        latent_size,
                    ]
                    samples = vq_model.decode_code(index_sample, qzshape)
                else:
                    qzshape = [
                        len(c_indices),
                        latent_size,
                        latent_size,
                        fsq_model.model.quantize.num_groups,
                    ]
                    samples = fsq_model.decode_code(index_sample, qzshape)
                samples = (
                    torch.clamp(127.5 * samples + 128.0, 0, 255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                images.append(samples)
            # Combine 8 images into a 2x4 grid
            # Stack all images into a large numpy array
            all_images = np.stack(
                [img[0] for img in images]
            )  # Take first image from each batch
            # Rearrange into 2x4 grid
            h, w = all_images.shape[1:3]
            grid = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
            for idx, image in enumerate(all_images):
                i, j = divmod(idx, 4)  # Calculate position in 2x4 grid
                grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = image

            # Save the combined image
            Image.fromarray(grid).save(
                f"{train_config['train']['output_dir']}/demo_samples.png"
            )

            return None
    else:
        cknna_scores = []
        for i in pbar:
            # Sample inputs:
            c_indices = torch.randint(
                0, train_config["data"]["num_classes"], (n,), device=device
            )

            gen_fun = generate_group if model.token_factorization else generate
            index_sample = gen_fun(
                model,
                c_indices,
                latent_size**2,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=True,
            )
            if vq_model is not None:
                qzshape = [
                    len(c_indices),
                    -1,
                    latent_size,
                    latent_size,
                ]
                samples = vq_model.decode_code(index_sample, qzshape)
                if eval_cknna:
                    model.train()
                    model.eval()
                    (_, _, layer_logits, factorized_ctx_layer_logits) = model(
                        cond_idx=c_indices,
                        idx=index_sample,
                        get_layer_logits=eval_cknna,
                    )
                    model.train()
                    model.eval()

                    dino_x = dino_transform(samples)
                    z = dino_model.forward_features(dino_x)
                    z = z["x_norm_patchtokens"].mean(1)
                    cknna_scores_per_layers = []
                    for lgs in layer_logits[::2]:
                        cknna_score = cknna(lgs.mean(1), z, topk=10)
                        cknna_scores_per_layers.append(cknna_score)
                    cknna_scores.append(cknna_scores_per_layers)
            else:
                qzshape = [
                    len(c_indices),
                    latent_size,
                    latent_size,
                    fsq_model.model.quantize.num_groups,
                ]
                samples = fsq_model.decode_code(index_sample, qzshape)
            if image_size_eval != image_size:
                samples = F.interpolate(
                    samples, size=(image_size_eval, image_size_eval), mode="bicubic"
                )
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * world_size + global_rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size
            dist.barrier()

        if len(cknna_scores) > 0:
            cknna_scores = np.array(cknna_scores).mean(0)
        if global_rank == 0:
            sample_folder_dir = create_npz_from_sample_folder(
                sample_folder_dir, train_config["sample"]["fid_num"]
            )
    return sample_folder_dir, cknna_scores


# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[LlamaGen-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = " ".join(map(str, messages))
    print(f"{prefix}: {combined_message}")


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--demo", action="store_true", default=False)
    args = parser.parse_args()
    train_config = load_config(args.config)
    # mixed_precision = train_config['sample']['precision'] if 'precision' in train_config['transport'] else 'no'

    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # get ckpt_dir
    assert "ckpt_path" in train_config, "ckpt_path must be specified in config"
    if global_rank == 0:
        print_with_prefix("Using ckpt:", train_config["ckpt_path"])
    ckpt_dir = train_config["ckpt_path"]

    fsq_model = None
    vq_model = None
    if "fsq" in train_config:
        fsq_model = FSQ_models["fsq"](
            train_config["fsq"]["config_path"],
            train_config["fsq"]["factorized_bits"],
        )
        checkpoint = torch.load(train_config["fsq"]["model_path"], map_location="cpu")[
            "ema_state_dict"
        ]
        checkpoint = {k.replace("module.", "model."): v for k, v in checkpoint.items()}
        msg = fsq_model.load_state_dict(checkpoint, strict=False)
        print(msg)
        fsq_model.to(device)
        fsq_model.eval()
    else:
        vq_model = VQ_models[train_config["vqvae"]["vq_model"]](
            codebook_size=train_config["vqvae"]["codebook_size"],
            codebook_embed_dim=train_config["vqvae"]["codebook_embed_dim"],
        )
        checkpoint = torch.load(train_config["vqvae"]["model_path"], map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        vq_model.to(device)
        vq_model.eval()

    if "vqvae" in train_config and "downsample_ratio" in train_config["vqvae"]:
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
    )
    model = Models[train_config["model"]["model_type"]](**kwargs)

    checkpoint = torch.load(ckpt_dir, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    elif "model" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()  # important!
    model.to(device)

    # naive sample
    sample_folder_dir, cknna_scores = do_sample(
        train_config,
        global_rank,
        world_size,
        ckpt_path=ckpt_dir,
        model=model,
        vq_model=vq_model,
        fsq_model=fsq_model,
        downsample_ratio=downsample_ratio,
        demo_sample_mode=args.demo,
    )
    print(sample_folder_dir, cknna_scores)
    if not args.demo:
        # calculate FID
        # Important: FID is only for reference, please use ADM evaluation for paper reporting
        if global_rank == 0:
            from tools.calculate_fid import calculate_fid_given_paths

            print_with_prefix(
                "Calculating FID with {} number of samples".format(
                    train_config["sample"]["fid_num"]
                )
            )
            assert (
                "fid_reference_file" in train_config["data"]
            ), "fid_reference_file must be specified in config"
            fid_reference_file = train_config["data"]["fid_reference_file"]
            fid = calculate_fid_given_paths(
                [fid_reference_file, sample_folder_dir],
                batch_size=50,
                dims=2048,
                device="cuda",
                num_workers=8,
                sp_len=train_config["sample"]["fid_num"],
            )
            print_with_prefix("fid=", fid)
