from lightning_modules import DeepfakeClassifier
from xai import generate_grad_cam_overlay
import torch
import torchvision.transforms as T
import gradio as gr
import spaces
import shutil
import gc
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT_PATH = "path/to/your/model/checkpoint.ckpt"

IMG_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
NUM_FRAMES_TO_PROCESS = 15

model = DeepfakeClassifier.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
model_name = model.hparams.model_name
model.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

def start_session(request: gr.Request):
    session_hash = request.session_hash
    session_dir = Path(f'/tmp/{session_hash}')
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Session with hash {session_hash} started.")
    return session_dir.as_posix()

def end_session(request: gr.Request):
    session_hash = request.session_hash
    session_dir = Path(f'/tmp/{session_hash}')
    
    if session_dir.exists():
        shutil.rmtree(session_dir)

    print(f"Session with hash {session_hash} ended.")

def extract_frames(video_filepath_temp: str, num_frames: int = 10):
    frames = []
    cap = cv2.VideoCapture(video_filepath_temp)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_filepath_temp}")
        return frames
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length < 1:
        cap.release()
        return frames
    
    actual_num_frames = min(num_frames, length)
    indices = np.linspace(0, length - 1, actual_num_frames).astype(int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None: continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

@spaces.GPU
def predict_deepfake(video_path, session_dir):
    if not video_path:
        gr.Error("No video uploaded.", duration=None)
        return [], []
    
    grad_cam_output = None
    target_layer = None
    original_path_name = Path(video_path).name
    video_name = Path(video_path).stem

    try:
        gr.Info(f"Loading video file: {video_name}", duration=2)
        frames = extract_frames(video_path, num_frames=NUM_FRAMES_TO_PROCESS)
    except Exception as load_e:
        gr.Error(f"Failed to load video file {original_path_name}: {load_e}", duration=None)

    try:
        model.to(device)
        model.to(torch.float32)
        gr.Info(f"Processing {original_path_name} on {device}...", duration=2)

        sequence_t_first = torch.stack([transform(frame) for frame in frames])
        sequence_t_first = sequence_t_first.to(device=device, dtype=torch.float32)
        sequence_c_first = sequence_t_first.permute(1, 0, 2, 3) # (C, T, H, W)
        batch_tensors = sequence_c_first.unsqueeze(0) # (B, T, C, H, W)
        
        with torch.no_grad():
            logits = model(batch_tensors)
            probs_fake = torch.softmax(logits, dim=1)[:, 1].item()
        
        predicted_label = "FAKE" if probs_fake >= 0.5 else "REAL"
        confidence = probs_fake if predicted_label == "FAKE" else 1 - probs_fake

        result = (
            f"**Prediction:** {predicted_label}\n"
            f"**Confidence:** {confidence:.2%}\n"
        )

        try:
            model_for_xai = model.model
            if (hasattr(model_for_xai, 'backbone')):
                target_layer = model_for_xai.backbone.conv4

            class_to_explain = 1 if predicted_label == "FAKE" else 0
                
            grad_cam_output = generate_grad_cam_overlay(
                model=model_for_xai,
                target_layer=target_layer,
                input_tensor=batch_tensors,
                original_frames=frames,
                target_class_idx=class_to_explain,
                img_size_for_overlay=IMG_SIZE,
            )
        except Exception as xai_e:
            gr.Error(f"Failed to generate XAI output: {xai_e}", duration=None)
            print(f"Error during XAI generation: {xai_e}")

        gr.Info("Prediction complete.", duration=2)
        return result, grad_cam_output

    except torch.cuda.OutOfMemoryError as e:
        error_msg = 'CUDA out of memory. Please try a shorter audio or reduce GPU load.'
        print(f"CUDA OutOfMemoryError: {e}")
        gr.Error(error_msg, duration=None)
    finally:
        try:
            if 'model' in locals() and hasattr(model, 'cpu'):
                if device == 'cuda':
                    model.cpu()
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as cleanup_e:
            print(f"Error during model cleanup: {cleanup_e}")
            gr.Warning(f"Issue during model cleanup: {cleanup_e}", duration=5)

article = (
    "<p style='font-size: 1.1em;'>"
    "This demo showcases a system for analyzing video authenticity, developed as part of a project for the "
    "Ethics in Artificial Intelligence course (Artificial Intelligence, University of Bologna)."
    "</p>"
    "<p><strong style='color: red; font-size: 1.2em;'>Key Features:</strong></p>"
    "<ul style='font-size: 1.1em;'>"
    "    <li>Classifies videos as REAL or FAKE</li>"
    "    <li>Provides an estimated probability for the prediction</li>"
    "    <li>Grad-CAM visualization to highlight image regions influencing the model's decision on a sample frame</li>"
    "</ul>"
    "<p style='text-align: center; margin-top: 1em;'>"
    "<p><strong>Disclaimer:</strong> This is a research and educational prototype. "
    "Results may vary and should not be considered definitive evidence of manipulation."
    "</p>"
)

examples = [
    # ["path/to/example_video1.mp4"],
]

with gr.Blocks() as demo:
    gr.Markdown(f"<h1 style='text-align: center; margin: 0 auto;'>👁️ Deepfake Video Detector</h1>")
    gr.HTML(article)

    current_video_path_state = gr.State(None)

    session_dir = gr.State()
    demo.load(start_session, outputs=[session_dir])

    with gr.Row():
        with gr.Column(scale=1, min_width=500):
            file_input = gr.Video(sources=["upload"], label="Upload Video File")
            gr.Examples(examples=examples, inputs=[file_input], label="Example Video Files (Click to Load)")
            predict_btn = gr.Button("Analyze Video", variant="primary")

        with gr.Column(scale=2):
            #gr.Markdown("---")
            xai_output = gr.Image(label="XAI Heatmap (e.g., Grad-CAM)", type="pil")
            gr.Markdown("<p><strong style='color: #FF0000; font-size: 1.2em;'>Prediction Results</strong></p>")
            results_output = gr.Markdown(label="Prediction Results", line_breaks=True)

    predict_btn.click(
        fn=predict_deepfake,
        inputs=[file_input, session_dir],
        outputs=[results_output, xai_output],
        api_name="predict_deepfake"
    )

    demo.unload(end_session)

if __name__ == "__main__":
    print("Launching Gradio Demo...")
    demo.queue()
    demo.launch()
