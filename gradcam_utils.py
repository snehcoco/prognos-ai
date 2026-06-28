"""
gradcam_utils.py — Grad-CAM implementation from scratch using PyTorch hooks.
No external grad-cam library required.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam(input_tensor, class_idx)   # class_idx=None → argmax
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        Returns: np.ndarray of shape (H, W) in [0, 1], where H, W = input_tensor spatial dims.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)           # (1, num_classes)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients → channel weights
        grads = self._gradients          # (1, C, H', W')
        acts  = self._activations        # (1, C, H', W')
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination + ReLU
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input spatial size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        return cam


def overlay_heatmap(image_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Blend Grad-CAM heatmap onto the original PIL image.
    Returns a PIL Image.
    """
    img_np = np.array(image_pil.convert("RGB"))
    h, w = img_np.shape[:2]

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (w, h))

    # Blend
    blended = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(blended)


def run_gradcam_pipeline(model, arch: str, input_tensor: torch.Tensor,
                          original_image: Image.Image, class_idx: int = None):
    """
    Full pipeline: build GradCAM, compute heatmap, overlay on original image.
    Returns: (overlay_pil, class_idx, cam_array)
    """
    from model_utils import get_gradcam_target_layer
    target_layer = get_gradcam_target_layer(model, arch)
    cam_obj = GradCAM(model, target_layer)
    try:
        cam = cam_obj(input_tensor, class_idx)
        if class_idx is None:
            with torch.no_grad():
                out = model(input_tensor)
            class_idx = out.argmax(dim=1).item()
        overlay = overlay_heatmap(original_image, cam)
    finally:
        cam_obj.remove_hooks()
    return overlay, class_idx, cam
