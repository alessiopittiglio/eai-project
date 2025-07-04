# Deepfake Video Detection: A Comparative Study

![Demo](./assets/demo.gif)

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

- Choose or create a YAML configuration file in `config/` (e.g., `config resnet_ffpp.yaml`).
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

### Running with nohup

To run the script in the background and ensure it continues executing even if the session is closed, use `nohup`:

```
nohup python src/train.py --config config/your_chosen_config.yaml > output.log 2>&1 &
```

- Replace `your_chosen_config.yaml` with the desired configuration file.
- The output will be logged to `output.log`. You can change the filename as needed.
- The `&` at the end runs the process in the background.

To check the process, use:

```
ps aux | grep train.py
```

To stop the process, use:

```
kill <process_id>
```

## Model benchmarks
This table summarizes the performance of different models.

| Model Name | Dataset                | Input Type      | Test Acc. | Test AUC | Checkpoint     | Config                                  | Date       |
|------------|------------------------|-----------------|-----------|----------|----------------|-----------------------------------------|------------|
| Xception3D | FF++ (c23, DF vs Orig) | Sequence (T=16) | 0.895     | 0.549    | [ Download ](https://drive.google.com/file/d/1JLvy7AzOIjjGiHes0JdFEZZdF1eOIEHo/view?usp=sharing) | [ View ]( config/xception3d_ffpp.yaml ) | 2025-05-15 |

## References

1. Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions* [Preprint]. arXiv. https://arxiv.org/abs/1610.02357
2. Westerlund, M. (2019). The Emergence of Deepfake Technology: A Review. *Technology Innovation Management Review*, *9*(11), 40–53. https://doi.org/10.22215/timreview/1282
3. Yan, Z., Zhang, Y., Yuan, X., Lyu, S., & Wu, B. (2023). *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection* [Preprint]. arXiv. https://arxiv.org/abs/2307.01426
