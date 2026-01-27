import torch
import numpy as np
import numpy.typing as npt
import math
from decord import VideoReader, cpu
from torch.nn import functional as F
from torchvision.transforms import Lambda, Compose
from torchvision.transforms._transforms_video import CenterCropVideo


def array_to_video(
    image_array: npt.NDArray, fps: float = 30.0, output_file: str = "output_video.mp4"
) -> None:
    """b h w c"""
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()


def custom_to_video(
    x: torch.Tensor, fps: float = 2.0, output_file: str = "output_video.mp4"
) -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 2, 3, 0).float().numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return


def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = 0
        e = s + sample_frames_len
        num_frames = num_frames
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_frames_len * num_frames)
        print(
            f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
            video_path,
            total_frames,
        )

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def tensor01_to_image(x):
    """[0-1] tensor to image"""
    x = x.detach().float().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.float().numpy()  # c h w
    x = (255 * x).astype(np.uint8)
    return x


def combined_image(img_list):
    # 确保图像列表不为空
    if len(img_list) == 0:
        raise ValueError("图像列表为空！")

    # 获取图像的维度（假设所有图像尺寸一致）
    c, h, w = img_list[0].shape

    # 计算合适的行列数
    num_images = len(img_list)
    cols = math.ceil(math.sqrt(num_images))  # 列数 = 图像总数的平方根，向上取整
    rows = math.ceil(num_images / cols)  # 行数 = 图像总数除以列数，向上取整

    # 将所有图片按行和列拼接
    images = np.concatenate(
        [
            np.concatenate(img_list[i * cols : (i + 1) * cols], axis=2)
            for i in range(rows)
        ],
        axis=1,
    )
    return images


def video_resize(x, resolution):
    height, width = x.shape[-2:]

    aspect_ratio = width / height
    if width <= height:
        new_width = resolution
        new_height = int(resolution / aspect_ratio)
    else:
        new_height = resolution
        new_width = int(resolution * aspect_ratio)
    resized_x = F.interpolate(
        x,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )
    return resized_x


def video_preprocess(video_data, short_size=128, crop_size=None):
    transform = Compose(
        [
            Lambda(lambda x: ((x / 255.0) * 2 - 1)),
            Lambda(lambda x: video_resize(x, short_size)),
            (
                CenterCropVideo(crop_size=crop_size)
                if crop_size is not None
                else Lambda(lambda x: x)
            ),
        ]
    )
    video_outputs = transform(video_data)
    # video_outputs = _format_video_shape(video_outputs)
    return video_outputs
