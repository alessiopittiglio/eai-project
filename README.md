# Deepfake Video Detection: A Comparative Study

## Usage

The primary workflow involves preprocessing raw videos into frames and then using configuration files with `src/train.py` for training.

### 1. Data preprocessing

If you haven't already processed the raw videos into frames, use the `scripts/preprocess_frames.py` script:

```
python scripts/preprocess_frames.py \
    --dataset <ffpp or dfdc> \
    --output_dir <path_to_save_processed_frames> \
    --num_frames <N> \
    [dataset_specific_arguments...] \
    [--skip_existing]
```

### 2. Training

Training is managed using src/train.py script, which loads its configuration from a specified YAML file:

- Choose or create a YAML configuration file in `config/` (e.g., `config ffpp_resnet18.yaml`).
- Run training script:

```
python src/train.py --config config/your_chosen_config.yaml
```
Example:
```
python src/train.py --config config/ffpp_resnet18.yaml
```

### (Optional) Using the script

For convenience, you can use the `run_experiment.sh` script:

- Edit the script to set dataset type, paths, and the CONFIG_FILE variable pointing to your desired YAML configuration.
- Run the following command:

```
chmod +x run_experiment.sh
./run_experiment.sh
```

## References

1. Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions* [Preprint]. arXiv. https://arxiv.org/abs/1610.02357
2. Westerlund, M. (2019). The Emergence of Deepfake Technology: A Review. *Technology Innovation Management Review*, *9*(11), 40–53. https://doi.org/10.22215/timreview/1282
3. Yan, Z., Zhang, Y., Yuan, X., Lyu, S., & Wu, B. (2023). *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection* [Preprint]. arXiv. https://arxiv.org/abs/2307.01426
