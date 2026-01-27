import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)

from ifsq.src.model.fsqvae import ImageFSQVAEModel
from torch import nn
import json
import numpy as np


class FSQModel(nn.Module):
    def __init__(self, config_path, factorized_bits=None):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        print(self.config)
        self.config["factorized_bits"] = factorized_bits
        print(self.config)
        self.model = ImageFSQVAEModel.from_config(self.config)

    def vocab_size(
        self,
    ):
        if self.config["factorized_bits"] is None:
            return [np.prod(self.config["levels"]).item()]
        else:
            sizes = []
            start, end = 0, 0
            for i in self.config["factorized_bits"]:  # [2, 2]
                end += i
                sub_size = np.prod(self.config["levels"][start:end]).item()
                sizes.append(sub_size)
                start = end
            return sizes

    def encode(self, x):

        h = self.model.encoder(x)
        if self.model.use_quant_layer:
            h = self.model.quant_conv(h)
        quant, indices = self.model.quantize(h)

        return quant, None, indices

    def decode(self, quant):
        if self.model.use_quant_layer:
            quant = self.model.post_quant_conv(quant)
        dec = self.model.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        if shape is not None:
            # print(f"{code_b.shape} to {shape}")
            code_b = code_b.reshape(shape)
        quant_b = self.model.quantize.indices_to_codes(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


FSQ_models = {"fsq": FSQModel}

if __name__ == "__main__":
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np

    config_path = (
        "ifsq/runs/fsq17x4_sig16/model.json"
    )
    model_path = "ifsq/results_noaug/fsq17x4_sig16-lr1.00e-04-bs16-rs256/checkpoint-160000.ckpt"
    # factorized_bits = [1, 3]
    factorized_bits = None
    model = FSQModel(config_path, factorized_bits)
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
        decoded_image = model.decode_code(indices)

    # Convert the output tensor to a PIL image and save it
    decoded_image = decoded_image.squeeze(0).cpu()  # Remove batch dimension
    decoded_image = decoded_image.permute(1, 2, 0)  # Convert to HWC format
    decoded_image = decoded_image * 0.5 + 0.5  # Unnormalize if necessary
    decoded_image = np.clip(decoded_image.numpy(), 0, 1)  # Ensure values are in [0, 1]

    # Save the decoded image
    decoded_image_pil = Image.fromarray((decoded_image * 255).astype(np.uint8))
    decoded_image_pil.save(image_path.replace(".jpg", "_rec.jpg"))
    print("Decoded image saved as 'decoded_image.jpg'")
