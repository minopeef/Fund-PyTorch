# Evaluation of Foundation Models for Earth Observation

This repository provides a comprehensive framework for evaluating various foundation models on Earth Observation tasks. The project is built with PyTorch Lightning and supports multiple state-of-the-art geospatial foundation models for both classification and segmentation tasks.

## Overview

This framework enables researchers and practitioners to:
- Evaluate multiple foundation models on standardized Earth Observation datasets
- Compare model performance across different tasks (classification, segmentation)
- Fine-tune models using various strategies (full fine-tuning, linear probing, LoRA)
- Conduct hyperparameter optimization using Ray Tune
- Manage experiments with Hydra configuration system

## Project Structure

```
Fund-PyTorch/
├── src/
│   ├── main.py                    # Main training script
│   ├── factory.py                 # Model and dataset factory functions
│   ├── configs/                   # Hydra configuration files
│   │   ├── config.yaml           # Main configuration
│   │   ├── model/                # Model-specific configurations
│   │   └── dataset/              # Dataset-specific configurations
│   ├── foundation_models/        # Foundation model implementations
│   │   ├── base.py               # Base model classes
│   │   ├── lightning_task.py     # PyTorch Lightning task base class
│   │   ├── dofa_wrapper.py       # DOFA model wrapper
│   │   ├── dinov2_wrapper.py     # DinoV2 model wrapper
│   │   ├── satmae_wrapper.py      # SatMAE model wrapper
│   │   ├── gfm_wrapper.py        # GFM model wrapper
│   │   ├── croma_wrapper.py      # CROMA model wrapper
│   │   ├── scalemae_wrapper.py   # ScaleMAE model wrapper
│   │   ├── senpamae_wrapper.py   # SenPaMAE model wrapper
│   │   ├── softcon_wrapper.py    # SoftCON model wrapper
│   │   └── anysat_wrapper.py     # AnySat model wrapper
│   ├── datasets/                  # Dataset wrappers
│   │   ├── data_module.py        # PyTorch Lightning data module
│   │   ├── geobench_wrapper.py   # GeoBench dataset wrapper
│   │   ├── benv2_wrapper.py      # BigEarthNetV2 wrapper
│   │   ├── resisc_wrapper.py     # RESISC45 wrapper
│   │   └── ...                   # Other dataset wrappers
│   ├── util/                      # Utility functions
│   ├── hparam_ray.py             # Ray Tune hyperparameter optimization
│   └── hparam_ray_hydra.py       # Ray Tune with Hydra integration
├── scripts/                       # Experiment scripts
│   ├── generate_bash_scripts.py  # Script generator for experiments
│   └── generate_bash_scripts_ray_tune.py  # Ray Tune script generator
├── tests/                         # Unit tests
├── requirements/                  # Dependency files
│   ├── required.txt              # Core dependencies
│   ├── style.txt                 # Code style dependencies
│   └── tests.txt                 # Test dependencies
├── pyproject.toml                # Project metadata and dependencies
└── README.md                     # This file
```

## Features

### Supported Foundation Models

- **CROMA**: Cross-modal foundation model for remote sensing
- **DOFA**: Dynamic Orthogonal Frequency Attention model
- **GFM**: Geospatial Foundation Model
- **DinoV2**: Vision Transformer from Meta AI
- **SatMAE**: Satellite Masked Autoencoder
- **ScaleMAE**: Scale-aware Masked Autoencoder
- **SenPaMAE**: Sentinel-2 Patch Masked Autoencoder
- **SoftCON**: Soft Contrastive Learning model
- **AnySat**: Any-scale satellite foundation model

### Supported Datasets

- **GeoBench**: Comprehensive geospatial benchmark suite including:
  - EuroSAT
  - ForestNet
  - So2Sat
  - Cashew
  - Chesapeake
  - Brick Kiln
  - NZ Cattle
  - NeonTree
  - PV4GER
  - Sacrop
- **BigEarthNetV2**: Large-scale multi-label classification dataset
- **RESISC45**: Remote sensing image scene classification dataset
- **Digital Typhoon**: Tropical cyclone dataset
- **Tropical Cyclone**: Additional cyclone dataset

### Training Strategies

- **Full Fine-tuning**: Update all model parameters
- **Linear Probing**: Freeze backbone, train only classification head
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with parameter-efficient methods
- **Selective Parameter Training**: Train only specific layers

## Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended)
- Conda or virtual environment manager

### Setup Instructions

1. Clone the repository and navigate to the root directory.

2. Create a conda environment:
```bash
conda create -n dofa-pytorch python=3.10 --yes
conda activate dofa-pytorch
```

3. Install OpenMIM and PyTorch:
```bash
pip install -U openmim
pip install torch==2.1.2
mim install mmcv==2.1.0 mmsegmentation==1.2.2
```

4. Install the package:
```bash
pip install -e .
```

### Optional: ViT Adapter Installation

The ViT Adapter module is optional and requires CUDA toolkit version less than 12. To install:

```bash
cd src/foundation_models/modules/ops/
sh make.sh
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
MODEL_WEIGHTS_DIR=<path/to/model/weights/directory>
TORCH_HOME=<path/to/torch/hub/weights>
DATASETS_DIR=<path/to/datasets/directory>
GEO_BENCH_DIR=<path/to/geobench/directory>
ODIR=<path/to/output/logs>
REPO_PATH=<path/to/this/repository>
```

If not set, `MODEL_WEIGHTS_DIR` defaults to `./fm_weights`. The framework will automatically download pre-trained model weights to this directory if they are not found.

### Model Weights

Pre-trained model weights are automatically downloaded from Hugging Face when first used. For SenPa-MAE, download weights manually and place them in the `MODEL_WEIGHTS_DIR` directory. Use `gdown` with the file ID from the Google Drive link.

## Usage

### Basic Training

Run an experiment using the main script:

```bash
export $(cat .env)
python src/main.py \
    output_dir=${ODIR}/exps/dinov2_cls_linear_probe_benv2_rgb \
    model=dinov2_cls_linear_probe \
    dataset=benv2_rgb \
    lr=0.002 \
    task=classification \
    num_gpus=1 \
    num_workers=8 \
    epochs=30 \
    warmup_epochs=5 \
    seed=13
```

### Configuration System

This project uses Hydra for experiment configuration. There are three types of configurations:

1. **Model Config**: Model-specific parameters (in `src/configs/model/`)
2. **Dataset Config**: Dataset-specific parameters (in `src/configs/dataset/`)
3. **Experiment Config**: Training parameters (can be overridden via command line)

You can override any configuration parameter using Hydra's override syntax:
- `model.{param_name}=value` for model parameters
- `dataset.{param_name}=value` for dataset parameters
- `{param_name}=value` for experiment parameters

### Generating Experiment Scripts

Use the convenience script to generate bash scripts for multiple experiments:

```bash
python scripts/generate_bash_scripts.py
```

This generates shell scripts in `scripts/<dataset>/run_<model>_<dataset>.sh` for each experiment configuration.

Run a generated script:
```bash
cd <path/to/repo>
sh scripts/<path/to/experiment>.sh
```

### Hyperparameter Tuning

For hyperparameter optimization with Ray Tune:

```bash
python scripts/generate_bash_scripts_ray_tune.py
```

This generates scripts that use `src/hparam_ray_hydra.py` to optimize learning rate and batch size. Additional Ray Tune parameters can be configured via `cfg.ray.{parameter_name}` in the command line.

## Adding New Models

To add a new foundation model:

1. **Create Model Wrapper**: Create a new file in `src/foundation_models/` (e.g., `new_model_wrapper.py`)

2. **Implement Base Class**: Inherit from `LightningTask` and implement required methods:
   - `forward()`: Model forward pass
   - `loss()`: Loss computation
   - `log_metrics()`: Metrics logging
   - `params_to_optimize()`: Parameters for optimization

3. **Register Model**: 
   - Import the model in `src/foundation_models/__init__.py`
   - Add to `model_registry` in `src/factory.py`

4. **Create Configuration**: Add a YAML config file in `src/configs/model/` following the naming convention `{model_name}_{task}_{strategy}.yaml`

## Adding New Datasets

To add a new dataset:

1. **Create Dataset Wrapper**: Create a new file in `src/datasets/` (e.g., `new_dataset_wrapper.py`)

2. **Implement Dataset Class**: Inherit from base dataset class and implement:
   - `create_dataset()`: Returns train, validation, and test datasets
   - Dataset-specific data loading logic

3. **Register Dataset**: Add to `dataset_registry` in `src/factory.py`

4. **Create Configuration**: Add a YAML config file in `src/configs/dataset/` with dataset-specific parameters

## Testing

Run unit tests to verify the installation and check for runtime errors:

```bash
pip install pytest
pytest tests/
```

Run tests for a specific model:
```bash
pytest tests/test_{model_name}.py
```

## Code Optimizations

The codebase has been optimized for:

- **Config Immutability**: Configuration objects are copied before modification to preserve original settings
- **Efficient Logging**: Replaced print statements with proper logging infrastructure
- **Error Handling**: Added checks for missing checkpoints and edge cases
- **Code Organization**: Removed work-in-progress files and improved code structure
- **Resource Management**: Proper handling of multi-GPU training and memory optimization

## Training Tips

### Multi-GPU Training

For distributed training with multiple GPUs:
```bash
python src/main.py \
    num_gpus=4 \
    strategy=ddp \
    ...
```

The learning rate is automatically scaled by the number of GPUs.

### LoRA Fine-tuning

To use LoRA for efficient fine-tuning:
```bash
python src/main.py \
    model=dofa_cls_lora \
    ...
```

LoRA configuration can be customized in the model config file.

### Resuming Training

Resume from a checkpoint:
```bash
python src/main.py \
    resume=<path/to/checkpoint.ckpt> \
    ...
```

## Output Structure

Experiments create the following directory structure:

```
output_dir/
├── checkpoints/
│   ├── best_model-epoch={epoch}.ckpt
│   └── last.ckpt
├── mlruns/
│   └── [MLFlow experiment tracking data]
└── [other experiment artifacts]
```

## Dependencies

Core dependencies include:
- PyTorch 2.1.2
- PyTorch Lightning
- Hydra (OmegaConf)
- Ray Tune (for hyperparameter optimization)
- MLFlow (for experiment tracking)
- Hugging Face Hub (for model weights)
- PEFT (for LoRA support)
- Various geospatial libraries (torchgeo, geobench)

See `pyproject.toml` and `requirements/required.txt` for complete dependency lists.

## Contributing

Contributions are welcome! When contributing:

1. Ensure all tests pass: `pytest tests/`
2. Follow the existing code style
3. Add tests for new features
4. Update documentation as needed
5. Submit pull requests with clear descriptions

## License

See LICENSE file for details.

## Notes

- The framework automatically handles model weight downloads from Hugging Face
- Some models may require specific CUDA versions or additional dependencies
- For large-scale experiments, consider using distributed training with multiple GPUs
- Hyperparameter tuning with Ray Tune requires additional computational resources
- The configuration system is flexible and allows for extensive customization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Weights Not Found**: Check `MODEL_WEIGHTS_DIR` environment variable
3. **Dataset Not Found**: Verify dataset paths in configuration files
4. **Import Errors**: Ensure all dependencies are installed and the package is installed in editable mode

### Getting Help

For issues and questions:
- Check existing issues in the repository
- Review configuration files for examples
- Consult model-specific documentation in respective subdirectories
