import os
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# NOTE: This script assumes that each dataset is organized in a standardized way:
# For image-based deepfake datasets (e.g., Celeb-DF, FaceForensics++), you need two 
# folders: "real" and "fake", containing images (or frames extracted from videos).
# For video datasets (e.g., DFDC), any frame extraction will need to be done beforehand.
# Modify the load_dataset functions according to your data's filesystem structure.

def download_ffpp(target_dir):
    """Download FaceForensics++ dataset (placeholder implementation)."""
    print(f"Downloading FaceForensics++ into {target_dir}...")
    # Placeholder: replace with actual download/extraction commands
    # os.system(f"wget -P {target_dir} <FFPP_DOWNLOAD_URL>")


def download_dfdc(target_dir):
    """Download DFDC dataset (placeholder implementation)."""
    print(f"Downloading DFDC into {target_dir}...")
    # Placeholder: replace with actual download/extraction commands
    # os.system(f"wget -P {target_dir} <DFDC_DOWNLOAD_URL>")


def download_celebd(target_dir):
    """Download Celeb-DF dataset (placeholder implementation)."""
    print(f"Downloading Celeb-DF into {target_dir}...")
    # Placeholder: replace with actual download/extraction commands
    # os.system(f"wget -P {target_dir} <CELEBDF_DOWNLOAD_URL>")


def download_dataset(name, base_dir):
    dataset_dir = Path(base_dir) / name 
    if dataset_dir.exists():
        print(f"Dataset {name} already exists in {dataset_dir}. Skipping download.")
        return
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if name.lower() in ['ffpp', 'faceforensics++']:
        download_ffpp(dataset_dir)
    elif name.lower() == 'dfdc':
        download_dfdc(dataset_dir)
    elif name.lower() in ['celebd', 'celeb-df']:
        download_celebd(dataset_dir)
    else:
        print(f"Download for dataset '{name}' not implemented. Please download manually.")


def load_dataset(name, base_dir):
    """Loads the dataset and returns a list of (path, label) tuples."""
    data = []
    dataset_dir = Path(base_dir) / name / 'test'
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found.")
    # Assumes 'REAL' and 'FAKE' subfolders containing image or video files
    for label_str, subfolder in [('real', 'REAL'), ('fake', 'FAKE')]:
        class_dir = dataset_dir / subfolder
        if not class_dir.exists():
            continue
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4')):
                    file_path = Path(root) / file
                    data.append((str(file_path), 0 if label_str == 'real' else 1))
                    
    print(len(data), "items loaded from dataset", name)
    return data


def subsample_data(data, percentage, seed=42):
    """
    Returns a subset of the data, balanced between real and fake, using the specified percentage.
    - data_list: list of (path, label) tuples
    - percentage: float between 0 and 100
    - seed: seed for reproducibility
    """
    if percentage <= 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100.")
    total = len(data)
    if total == 0:
        return []
    # Separare per label
    reals = [item for item in data if item[1] == 0]
    fakes = [item for item in data if item[1] == 1]
    desired_total = int(total * (percentage / 100.0))
    desired_each = desired_total // 2
    max_each = min(len(reals), len(fakes))
    desired_each = min(desired_each, max_each)
    if desired_each == 0 and max_each > 0 and percentage > 0:
        desired_each = 1
    random.seed(seed)
    sampled_reals = random.sample(reals, desired_each) if reals else []
    sampled_fakes = random.sample(fakes, desired_each) if fakes else []
    subset = sampled_reals + sampled_fakes
    random.shuffle(subset)
    return subset


class DeepfakeImageDataset(Dataset):
    """Custom PyTorch Dataset for deepfake images."""
    def __init__(self, data_list, processor):
        # data_list: list of (path, label)
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        image = Image.open(path).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')
        # inputs: dictionary with 'pixel_values'
        return inputs['pixel_values'].squeeze(0), label


def load_model(model_name, device):
    """Loads a Hugging Face model for deepfake classification (images or sequences)."""
    # Try image classification model
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        return {
            'type': 'image-classification',
            'model': model.to(device),
            'processor': feature_extractor
        }
    except Exception:
        pass
    # Try sequence classification model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {
            'type': 'sequence-classification',
            'model': model.to(device),
            'processor': tokenizer
        }
    except Exception:
        pass
    # Fallback: pipeline per image-classification
    try:
        pipe = pipeline('image-classification', model=model_name, device=0 if device == 'cuda' else -1)
        return {
            'type': 'pipeline',
            'pipeline': pipe
        }
    except Exception as e:
        raise ValueError(f"Unable to load model {model_name}: {e}")


def compute_metrics(preds, labels):
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds)
    metrics['recall'] = recall_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds)
    try:
        metrics['roc_auc'] = roc_auc_score(labels, preds)
    except ValueError:
        metrics['roc_auc'] = None
    return metrics


def evaluate_image_model(model_dict, data, device, batch_size=16):
    """Evaluates an image-classification model using batching."""
    model = model_dict['model']
    processor = model_dict['processor']
    dataset = DeepfakeImageDataset(data, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating image model (batches)"):
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            preds_batch = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().tolist()
            all_preds.extend(preds_batch)
            all_labels.extend(labels.cpu().tolist())
    return all_preds, all_labels


def evaluate_pipeline_model(model_dict, data, batch_size=16):
    """Evaluates a Hugging Face Pipeline for image-classification in batches."""
    pipe = model_dict['pipeline']
    all_preds = []
    all_labels = []
    # Create a list of all paths and corresponding labels
    paths = [item[0] for item in data]
    labels = [item[1] for item in data]
    # Process in batches
    for i in tqdm(range(0, len(paths), batch_size), desc="Evaluating pipeline (batches)"):
        batch_paths = paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        results = pipe(batch_paths)
       # results: list where each element is a list of dictionaries [{ 'label': ..., 'score': ... }, ...]
        for output_list, lbl in zip(results, batch_labels):
            if isinstance(output_list, list) and len(output_list) > 0:
                label_str = output_list[0]['label'].lower()
            else:
                # If unexpected format, take the dictionary directly
                label_str = output_list['label'].lower() if isinstance(output_list, dict) else ''
            pred_label = 1 if ('1' in label_str or 'fake' in label_str) else 0
            all_preds.append(pred_label)
            all_labels.append(lbl)
    return all_preds, all_labels


def evaluate_model_on_dataset(model_dict, dataset_name, base_dir, device, percentage, batch_size):
    # Load and subsample the dataset
    data_full = load_dataset(dataset_name, base_dir)
    data = subsample_data(data_full, percentage, seed=42)
    if not data:
        raise ValueError(f"No data available for dataset {dataset_name} with the specified percentage.")
    # Perform evaluation based on model type
    if model_dict['type'] == 'image-classification':
        preds, labels = evaluate_image_model(model_dict, data, device, batch_size)
    elif model_dict['type'] == 'pipeline':
        preds, labels = evaluate_pipeline_model(model_dict, data, batch_size)
    else:
        raise NotImplementedError(f"Evaluation for model type {model_dict['type']} not implemented.")
    metrics = compute_metrics(preds, labels)
    metrics['num_samples'] = len(data)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face models for deepfakes on multiple datasets.")
    parser.add_argument('--models', nargs='+', required=True, help="List of Hugging Face model names or paths.")
    parser.add_argument('--datasets', nargs='+', required=True, help="List of dataset names (e.g., ffpp, dfdc, celebd).")
    parser.add_argument('--data-dir', type=str, default='datasets', help="Base directory to store/download datasets.")
    parser.add_argument('--download', action='store_true', help="If set, download datasets before evaluation.")
    parser.add_argument('--test-percentage', type=float, default=100.0, help="Percentage of the dataset to use for testing (balanced). Use 100 for the full set.)")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument('--output-json', type=str, default='results.json', help="Path for the output JSON file with results.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device: 'cuda' or 'cpu'.")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    results = {}
    for dset in args.datasets:
        if args.download:
            download_dataset(dset, args.data_dir)

    for model_name in args.models:
        print(f"Loading model {model_name}...")
        try:
            model_dict = load_model(model_name, args.device)
        except ValueError as e:
            print(e)
            continue
        results[model_name] = {}
        for dset in args.datasets:
            print(f"Evaluating mode {model_name} on dataset {dset} ({args.test_percentage}% of data), batch size {args.batch_size}...")
            try:
                metrics = evaluate_model_on_dataset(
                    model_dict,
                    dset,
                    args.data_dir,
                    args.device,
                    args.test_percentage,
                    args.batch_size
                )
                results[model_name][dset] = metrics
            except Exception as e:
                print(f"Error evaluating {model_name} su {dset}: {e}")
                results[model_name][dset] = {'error': str(e)}

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to  {args.output_json}")

if __name__ == '__main__':
    main()
