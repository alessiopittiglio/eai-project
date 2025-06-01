import os
import yaml
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import ImageFile, Image
import gradio as gr

from transformers import AutoModelForImageClassification
from timm.data.transforms_factory import create_transform

from lightning_modules import DeepFakeFinetuningLightningModule

# Allow PIL to load truncated images without raising
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── STEP 1 ── Load your config.yaml ─────────────────────────────────────────────
CONFIG_PATH="/cluster/home/robergio/ondemand/giordanopittiglio2425/outputs/MambaVision/version_7/config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# cfg should contain at least:
#   data_dir: path/to/your/dataset_root
#   model_name: "nvidia/MambaVision-T-1K"
#   num_classes: 2
#   backbone_lr, head_lr, weight_decay (unused at inference)
#   image_size, batch_size, num_workers
#   downsample / upsample flags (unused at inference)
#   max_epochs (unused)
#   monitor_metric, mode (unused at inference)

DATA_DIR = cfg["data_dir"]  # e.g. "/path/to/dataset"
CKPT_PATH = "/cluster/home/robergio/ondemand/giordanopittiglio2425/outputs/MambaVision/version_7/best-epoch=09-val_loss=0.3672.ckpt"
class_counts = cfg.get("class_counts", None)


# ── STEP 4 ── Instantiate & load state_dict from your checkpoint ────────────────────

# 4.1) Instantiate the LightningModule with the same cfg & class_counts
lit_model = DeepFakeFinetuningLightningModule(cfg, class_counts)

# 4.2) Load weights
checkpoint = torch.load(CKPT_PATH, map_location="cpu")
# Lightning serializes the state_dict under "state_dict"
lit_model.load_state_dict(checkpoint["state_dict"])
lit_model.eval()

# Move to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model.to(device)


# ── STEP 5 ── Reconstruct the exact “validation” transforms ────────────────────────

# We need to apply the same center‐crop / normalize that you used during training/validation.
tmp = AutoModelForImageClassification.from_pretrained(
    cfg["model_name"], trust_remote_code=True
)
mean = tmp.config.mean
std = tmp.config.std
crop_mode = tmp.config.crop_mode
crop_pct = tmp.config.crop_pct
tmp.cpu()
del tmp

val_transform = create_transform(
    input_size=(3, cfg["image_size_width"], cfg["image_size_height"]),
    is_training=False,
    mean=mean,
    std=std,
    crop_mode=crop_mode,
    crop_pct=crop_pct,
)


# ── STEP 6 ── Build a helper to map index→label_name (REAL/FAKE) using your dataset's class_to_idx ─

val_folder = os.path.join(DATA_DIR, "val")
val_dataset = ImageFolder(val_folder)  # we only need it to get class_to_idx
idx2class = {v: k for k, v in val_dataset.class_to_idx.items()}
# e.g. idx2class = {0:"FAKE", 1:"REAL"}


# ── STEP 7 ── Define the Gradio prediction function ─────────────────────────────────

def predict_image(pil_img: Image.Image):
    """
    Input: a PIL image
    Output: a dict { "REAL": prob_real, "FAKE": prob_fake }
    """
    img = pil_img.convert("RGB")
    x = val_transform(img).unsqueeze(0)  # [1,3,H,W]
    x = x.to(device)

    with torch.no_grad():
        logits = lit_model(x)             # [1, 2]
        probs = F.softmax(logits, dim=1)  # [1, 2]
        probs = probs.cpu().squeeze(0)    # [2]

    # Build a {label: probability} dictionary
    out = {
        idx2class[0]: float(probs[0]),  # e.g. "FAKE": 0.73
        idx2class[1]: float(probs[1]),  # e.g. "REAL": 0.27
    }
    return out


# ── STEP 8 ── Launch the Gradio Interface ───────────────────────────────────────────

if __name__ == "__main__":
    # Use gradio.Label to show top‐2 probabilities
    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=2),
        title="DeepFake Detection",
        description="Upload an image → Model predicts REAL vs FAKE (with probabilities).",
    )

    iface.launch(server_name="0.0.0.0", server_port=7860)
