import os
from typing import Any, Dict, List, Union

import pytorch_lightning
import torch.nn.functional
import torch.optim
import torchvision.datasets
import torchvision.transforms
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

Batch = List[torch.Tensor]
Result = Dict[str, Union[torch.Tensor, Any]]


class System(LightningModule):

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(28 * 28, 10)
        self.dataset_train = None
        self.dataset_valid = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x.view(x.size(0), -1)).relu()

    def training_step(self, batch: Batch, batch_idx: int) -> Result:
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch: Batch, batch_idx: int) -> Result:
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': logs}

    def test_step(self, batch: Batch, batch_idx: int) -> Result:
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return {'test_loss': loss}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def train_val_dataloader(self) -> List[torch.utils.data.Dataset]:
        if self.dataset_train is not None:
            return
        dataset = torchvision.datasets.MNIST(
                os.getcwd(),
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor())
        n = len(dataset)
        n_train = int(0.9 * n)
        n_valid = n - n_train
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset, (n_train, n_valid))
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid

    @pytorch_lightning.data_loader
    def train_dataloader(self) -> DataLoader:
        self.train_val_dataloader()
        return DataLoader(self.dataset_train, batch_size=32)

    @pytorch_lightning.data_loader
    def val_dataloader(self) -> DataLoader:
        self.train_val_dataloader()
        return DataLoader(self.dataset_valid, batch_size=32)

    @pytorch_lightning.data_loader
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
                torchvision.datasets.MNIST(
                    os.getcwd(),
                    train=False,
                    download=True,
                    transform=torchvision.transforms.ToTensor()),
                batch_size=32)


model = System()
trainer = Trainer(min_epochs=10)
trainer.fit(model)
