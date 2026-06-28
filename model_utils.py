"""
model_utils.py — Load PyTorch model checkpoints for inference.

ALL three models were trained with timm.create_model() in Colab, so we
reconstruct them the same way here to guarantee state-dict compatibility.

  timm model names used during training:
    EfficientNet-B0 → "efficientnet_b0"
    ResNet50        → "resnet50"
    DenseNet121     → "densenet121"

Grad-CAM target layers (from training notebook):
    EfficientNet-B0 → model.conv_head
    ResNet50        → model.layer4[-1]
    DenseNet121     → model.features.norm5
"""

import torch

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# Internal timm model name for each architecture label
_TIMM_NAMES = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet50":        "resnet50",
    "DenseNet121":     "densenet121",
}


def load_model(arch: str, num_classes: int, ckpt_path: str):
    """
    Build the architecture with timm, load the state dict, return eval-mode model.
    Checkpoint must be a plain state_dict (OrderedDict), as saved during training.
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Run: pip install timm")
    if arch not in _TIMM_NAMES:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(_TIMM_NAMES)}")

    model = timm.create_model(_TIMM_NAMES[arch], pretrained=False, num_classes=num_classes)

    state = torch.load(ckpt_path, map_location="cpu")
    # Support wrapped checkpoints {"model_state_dict": ...} or {"state_dict": ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def get_gradcam_target_layer(model, arch: str):
    """Return the target layer for Grad-CAM, matching what the training notebook used."""
    if arch == "EfficientNet-B0":
        return model.conv_head          # 1280-ch conv, final feature map
    elif arch == "ResNet50":
        return model.layer4[-1]         # Last residual block
    elif arch == "DenseNet121":
        return model.features.norm5     # Final BatchNorm after all dense blocks
    else:
        raise ValueError(f"No Grad-CAM target defined for '{arch}'")
