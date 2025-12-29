import datetime
import os
from pathlib import Path
import warnings
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from datasets.data_module import BenchmarkDataModule
from lightning.pytorch import seed_everything
from factory import create_model
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """Calculate and log trainable parameters statistics."""
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_percent = 100 * trainable_params / all_param if all_param > 0 else 0.0
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {trainable_percent:.2f}"
    )


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Seed everything
    seed_everything(cfg.seed)

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Scale learning rate for multi-GPU (create a copy to avoid modifying original config)
    scaled_lr = cfg.lr * cfg.num_gpus

    # Setup logger
    experiment_name = f"{cfg.model.model_type}_{cfg.dataset.dataset_name}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_run_{timestamp}",
        tracking_uri=f"file:{os.path.join(cfg.output_dir, 'mlruns')}",
    )

    # Callbacks
    model_monitor = "val_miou" if cfg.task == "segmentation" else "val_acc1"
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model-{epoch}",
            monitor=model_monitor,
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer_kwargs = {
        "logger": mlf_logger,
        "callbacks": callbacks,
        "devices": cfg.num_gpus,
        "max_epochs": cfg.epochs,
        "num_sanity_val_steps": 0,
    }
    
    if cfg.strategy == "ddp" and cfg.num_gpus > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
    
    trainer = Trainer(**trainer_kwargs)

    # Initialize data module (create a copy of dataset config to avoid modifying original)
    dataset_config = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_config.image_resolution = cfg.model.image_resolution
    data_module = BenchmarkDataModule(
        dataset_config=dataset_config,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
    )

    # Create model (assumed to be a LightningModule)
    # Update lr in args for model creation
    args_config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    args_config.lr = scaled_lr
    model = create_model(args_config, cfg.model, dataset_config)

    print_trainable_parameters(model)

    # Train
    ckpt_path = cfg.resume if cfg.resume else None
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Test
    best_checkpoint_path = callbacks[0].best_model_path
    if best_checkpoint_path:
        trainer.test(model, data_module, ckpt_path=best_checkpoint_path)
    else:
        logger.warning("No best checkpoint found, skipping test evaluation")


if __name__ == "__main__":
    # Set default environment variables if not already set
    if "MODEL_WEIGHTS_DIR" not in os.environ:
        os.environ["MODEL_WEIGHTS_DIR"] = "./fm_weights"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    main()
