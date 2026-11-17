"""Model definitions for MiniMind-O Action.

Exposes the 63.9M Dense language backbone, the frozen SigLIP2 vision encoder
wrapper, the trainable cross-modal projector, and the GUI action model.
"""
from .minimind import MiniMindConfig, MiniMindForCausalLM, MiniMindModel
from .vision_interface import prepare_frozen_vision
from .vision import (
    MMVisionProjector, FrozenSigLIP2, GuiActionModel,
    inject_vision_features, preprocess_rgb, lm_hidden_forward, build_backbone,
    IMAGE_ID, IMAGE_TOKEN, TOKENS_PER_IMAGE, IN_DIM, IMG_SIZE,
)
