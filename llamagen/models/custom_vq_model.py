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

sys.path.append("../..")
from ifsq.src.model.vqvae.modeling_imagevqvae import ImageVQVAEModel


class VQModel(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        print(self.config)
        self.model = ImageVQVAEModel.from_config(self.config)

    def encode(self, x):
        h = self.model.encoder(x)
        if self.model.use_quant_layer:
            h = self.model.quant_conv(h)
        quant, indices, _ = self.model.quantize(h)
        return quant, None, (None, None, indices)

    def decode(self, quant):
        if self.model.use_quant_layer:
            quant = self.model.post_quant_conv(quant)
        dec = self.model.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.model.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


CusVQ_models = {"cusvq": VQModel}
if __name__ == "__main__":
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np

    config_path = (
        "ifsq/runs/vq_dim8/model.json"
    )
    model_path = "ifsq/results_noaug/vq_dim8-lr1.00e-04-bs16-rs256/checkpoint-5000.ckpt"

    model = VQModel(config_path)
    checkpoint = torch.load(model_path, map_location="cpu")["ema_state_dict"]
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint = {"model." + k: v for k, v in checkpoint.items()}
    msg = model.load_state_dict(checkpoint, strict=False)
    print(model)
    print(msg)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Total parameters: {sum(p.numel() for p in model.model.encoder.parameters())}"
    )
    print(
        f"Total parameters: {sum(p.numel() for p in model.model.decoder.parameters())}"
    )

    # Load and process an image
    image_path = "assets/logo.jpg"  # Update with your image path
    image = Image.open(image_path).convert("RGB")

    # Define image transformations: resize, tensor conversion, and normalization
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to fit model input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize if required
        ]
    )

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image tensor to the appropriate device (e.g., GPU if available)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)

    # Encode and decode the image
    with torch.no_grad():
        quant, diff, indices = model.encode(image_tensor)
        print(indices.shape)
        print(indices)
        # decoded_image = model.decode(quant)
        decoded_image = model.decode_code(indices, shape=(1, 8, 32, 32))

    # Convert the output tensor to a PIL image and save it
    decoded_image = decoded_image.squeeze(0).cpu()  # Remove batch dimension
    decoded_image = decoded_image.permute(1, 2, 0)  # Convert to HWC format
    decoded_image = decoded_image * 0.5 + 0.5  # Unnormalize if necessary
    decoded_image = np.clip(decoded_image.numpy(), 0, 1)  # Ensure values are in [0, 1]

    # Save the decoded image
    decoded_image_pil = Image.fromarray((decoded_image * 255).astype(np.uint8))
    decoded_image_pil.save(image_path.replace(".jpg", "_rec.jpg"))
    print("Decoded image saved as 'decoded_image.jpg'")
