"""Base Class for Lightning Tasks."""

from lightning import LightningModule
import torch
from typing import Any, Dict, Tuple


class LightningTask(LightningModule):
    """Base class for all foundation model tasks in PyTorch Lightning.
    
    This class provides common functionality for classification, regression,
    and segmentation tasks with foundation models.
    """
    
    def __init__(self, args: Any, model_config: Any, data_config: Any):
        """Initialize the Lightning task.
        
        Args:
            args: Training arguments/configuration.
            model_config: Model-specific configuration.
            data_config: Dataset-specific configuration.
        """
        super().__init__()
        self.model_config = model_config
        self.args = args
        self.data_config = data_config
        self.save_hyperparameters()

    def loss(self, outputs, labels):
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def freeze(self, module):
        """Freeze all parameters in the given module."""
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module):
        """Unfreeze all parameters in the given module."""
        for param in module.parameters():
            param.requires_grad = True

    def freeze_non_lora_params(self, module):
        """
        Freeze the encoder parameters except for LoRA-specific ones.
        """
        for name, param in module.named_parameters():
            if "lora" not in name:  # Skip LoRA parameters
                param.requires_grad = False

    def log_metrics(self, outputs, targets, prefix="train"):
        """Abstract method for logging task-specific metrics."""
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def forward(self, samples):
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for a single batch.
        
        Args:
            batch: Tuple of (images, targets).
            batch_idx: Index of the current batch.
            
        Returns:
            Computed loss value.
        """
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="train")
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch.
        
        Args:
            batch: Tuple of (images, targets).
            batch_idx: Index of the current batch.
            
        Returns:
            Computed loss value.
        """
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="val")
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step for a single batch.
        
        Args:
            batch: Tuple of (images, targets).
            batch_idx: Index of the current batch.
            
        Returns:
            Computed loss value.
        """
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="test")
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with cached dataloader length."""
        if self.model_config.task in ["classification", "regression"]:
            optimizer = torch.optim.SGD(
                self.params_to_optimize(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(self.params_to_optimize(), lr=self.args.lr)

        # Cache dataloader length to avoid multiple calls
        train_dataloader = self.trainer.datamodule.train_dataloader()
        dataloader_length = len(train_dataloader)
        
        if self.args.num_gpus >= 1:
            num_warmup_steps = (
                dataloader_length * self.args.warmup_epochs // self.args.num_gpus
            )
            total_steps = (
                dataloader_length * self.args.epochs // self.args.num_gpus
            )
        else:
            num_warmup_steps = dataloader_length * self.args.warmup_epochs
            total_steps = dataloader_length * self.args.epochs

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.lr,
            total_steps=total_steps,
            anneal_strategy="cos",  # Cosine annealing
            pct_start=float(num_warmup_steps) / float(total_steps) if total_steps > 0 else 0.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
