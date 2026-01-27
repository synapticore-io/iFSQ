from .llamagen import *
from .fsq_model import FSQ_models
from .generate import generate
from .vq_model import VQ_models
from .custom_vq_model import CusVQ_models
from .generate_group import generate_group

Models = {}
Models.update(LlamaGen_models)
