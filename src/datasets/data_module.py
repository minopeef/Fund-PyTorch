from lightning import LightningDataModule
import torch
from src.factory import create_dataset


class BenchmarkDataModule(LightningDataModule):
    """Optimized data module for benchmark datasets with reduced code duplication."""
    
    def __init__(self, dataset_config, batch_size, num_workers, pin_memory):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._datasets = None

    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing."""
        if self._datasets is None:
            self._datasets = create_dataset(self.dataset_config)
        self.dataset_train, self.dataset_val, self.dataset_test = self._datasets

    def _create_dataloader(self, dataset, shuffle=False, drop_last=False):
        """Helper method to create dataloaders with common configuration."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        """Create training dataloader."""
        return self._create_dataloader(
            self.dataset_train,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return self._create_dataloader(
            self.dataset_val,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return self._create_dataloader(
            self.dataset_test,
            shuffle=False,
            drop_last=False,
        )
