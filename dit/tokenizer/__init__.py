from .marvae import MAR_VAE
from .vavae import VA_VAE
from .sdvae import SD_VAE
from .fsq import FSQ
from .cus_ae import Cus_AE

VAE_Models = {
    "marvae": MAR_VAE,
    "vavae": VA_VAE,
    "sdvae": SD_VAE,
    "fsq": FSQ,
    "cus_ae": Cus_AE,
}
