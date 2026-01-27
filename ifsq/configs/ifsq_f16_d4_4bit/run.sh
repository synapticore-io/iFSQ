
export WANDB_PROJECT='iFSQ'
export WANDB_MODE="offline"
export WANDB_API_KEY=""

IMAGENETT_TRAIN="path/to/your/imagenet/train"
IMAGENETT_VAL="path/to/your/imagenet/val"
COCO2017_VAL="path/to/your/coco2017/val"

torchrun --nproc_per_node=8 \
    train_ddp.py \
    --exp_name ifsq_f16_d4_4bit \
    --image_path ${IMAGENETT_TRAIN} \
    --imgnet_eval_path ${IMAGENETT_VAL} \
    --coco_eval_path ${COCO2017_VAL} \
    --model_name ImageFSQVAE \
    --model_config configs/ifsq_f16_d4_4bit/run.json \
    --resolution 256 \
    --batch_size 16 \
    --dataset_num_worker 8 \
    --lr 0.0001 \
    --weight_decay 0.0 \
    --epochs 100 \
    --disc_start 50000 \
    --save_ckpt_step 5000 \
    --eval_steps 5000 \
    --eval_batch_size 64 \
    --eval_subset_size 50000 \
    --eval_lpips \
    --eval_psnr \
    --eval_ssim \
    --eval_fid \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --kl_weight 0.0 \
    --loss_type l1 \
    --disc_cls src.model.losses.LPIPSWithDiscriminator2D