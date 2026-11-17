"""Explicit placement for Omni's externally held frozen vision encoder."""
import torch

def prepare_frozen_vision(omni, device):
    """Move and freeze the existing encoder; never download or load weights.

    MiniMindOmni stores the encoder outside registered children. Therefore
    ``omni.to(device)`` alone does not move it. The projector remains trainable.
    """
    encoder = getattr(omni, 'vision_encoder', None)
    projector = getattr(omni, 'vision_proj', None)
    if not isinstance(encoder, torch.nn.Module) or not isinstance(projector, torch.nn.Module):
        raise ValueError('vision encoder or projector missing')
    device = torch.device(device)
    encoder.to(device); encoder.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    params=list(projector.parameters())
    if not params or not any(p.requires_grad for p in params):
        raise ValueError('vision projector has no trainable parameters')
    if any(p.device != device for p in params):
        raise ValueError('move the Omni model and projector to the target device first')
    if any(p.device != device or p.requires_grad for p in encoder.parameters()):
        raise ValueError('encoder device or freeze contract failed')
    return {'encoder_type':type(encoder).__name__,'device':str(device),
            'encoder_frozen':True,'projector_trainable_parameters':sum(p.numel() for p in params if p.requires_grad)}
