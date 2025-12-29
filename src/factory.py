"""Factory utility functions to create datasets and models."""

import logging
from src.foundation_models import (
    CromaModel,
    ScaleMAEModel,
    GFMModel,
    DinoV2Model,
    SoftConModel,
    DofaModel,
    SatMAEModel,
    AnySatModel,
    SenPaMAEModel,
)
from src.datasets.geobench_wrapper import GeoBenchDataset
from src.datasets.resisc_wrapper import Resics45Dataset
from src.datasets.benv2_wrapper import BenV2Dataset
from src.datasets.digital_typhoon_wrapper import DigitalTyphoonDataset
from src.datasets.tropical_cyclone_wrapper import TropicalCycloneDataset
from src.datasets.dummy_dataset import DummyWrapper

logger = logging.getLogger(__name__)

model_registry = {
    "croma": CromaModel,
    "scalemae": ScaleMAEModel,
    "gfm": GFMModel,
    "dinov2": DinoV2Model,
    "softcon": SoftConModel,
    "dofa": DofaModel,
    "satmae": SatMAEModel,
    "anysat": AnySatModel,
    "senpamae": SenPaMAEModel,
}

dataset_registry = {
    "geobench": GeoBenchDataset,
    "resisc45": Resics45Dataset,
    "benv2": BenV2Dataset,
    "digital_typhoon": DigitalTyphoonDataset,
    "tropical_cyclone": TropicalCycloneDataset,
    "dummy": DummyWrapper,
}


def create_dataset(config_data):
    """Create dataset splits for training, validation, and testing.
    
    Args:
        config_data: Dataset configuration object with dataset_type attribute.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
        
    Raises:
        ValueError: If dataset_type is not found in registry.
        AttributeError: If config_data lacks dataset_type attribute.
    """
    if not hasattr(config_data, "dataset_type"):
        raise AttributeError("config_data must have 'dataset_type' attribute")
    
    dataset_type = config_data.dataset_type
    if not dataset_type:
        raise ValueError("dataset_type cannot be empty")
    
    dataset_class = dataset_registry.get(dataset_type.lower())
    if dataset_class is None:
        available = ", ".join(dataset_registry.keys())
        raise ValueError(
            f"Dataset type '{dataset_type}' not found. "
            f"Available types: {available}"
        )
    
    logger.debug(f"Creating dataset: {dataset_type}")
    dataset = dataset_class(config_data)
    return dataset.create_dataset()


def create_model(args, config_model, dataset_config=None):
    """Create a model instance based on configuration.
    
    Args:
        args: Training arguments/configuration.
        config_model: Model configuration with model_type attribute.
        dataset_config: Optional dataset configuration.
        
    Returns:
        Model instance (LightningModule).
        
    Raises:
        ValueError: If model_type is not found in registry.
        AttributeError: If config_model lacks model_type attribute.
    """
    if not hasattr(config_model, "model_type"):
        raise AttributeError("config_model must have 'model_type' attribute")
    
    model_name = config_model.model_type
    if not model_name:
        raise ValueError("model_type cannot be empty")
    
    model_class = model_registry.get(model_name.lower())
    if model_class is None:
        available = ", ".join(model_registry.keys())
        raise ValueError(
            f"Model type '{model_name}' not found. "
            f"Available types: {available}"
        )
    
    logger.debug(f"Creating model: {model_name}")
    model = model_class(args, config_model, dataset_config)
    return model
