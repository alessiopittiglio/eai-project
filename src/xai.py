import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from typing import List

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM

def generate_grad_cam_overlay(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor, # (B, C, H, W)
    original_frames: List[Image.Image],
    target_class_idx: int,
    img_size_for_overlay: int,
    cam_method: type = GradCAM,
    temporal_aggregation_method: str = "mean",
    target_frame_idx: int = -1,
):
    """
    Generates a Grad-CAM heatmap and overlays it on the original image.
    Returns a PIL Image of the overlay or None on error.
    """
    with cam_method(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam_3d_batch = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam_3d = grayscale_cam_3d_batch[0, :, :, :]

        if temporal_aggregation_method == "mean":
            aggregated_heatmap_2d = np.mean(grayscale_cam_3d, axis=0)

        if target_frame_idx == -1 or not (0 <= target_frame_idx < len(original_frames)):
            idx_representative_frame = len(original_frames) // 2
        else:
            idx_representative_frame = target_frame_idx

        representative_frame = original_frames[idx_representative_frame]
        rgb_image = np.array(representative_frame) / 255.0

        target_h, target_w = rgb_image.shape[:2]
        heatmap_resized = cv2.resize(
            aggregated_heatmap_2d, 
            (target_w, target_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        visualization = show_cam_on_image(
            rgb_image, 
            heatmap_resized, 
            use_rgb=True, 
            image_weight=0.5
        )

        return Image.fromarray(visualization.astype(np.uint8))
