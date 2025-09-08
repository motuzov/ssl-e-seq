import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from seq_loader.names_loader import (
    TDTSDataset,
    collate_padded_batch_fn,
    PaddedTDTSDBatch,
    ColEmbeddingsParams,
)

from seq_loader.nn import CatColumnsDataEncoder, TBDTSLstm
from torchmetrics.classification import MulticlassAccuracy
import lightning as L


class LitLSTM(L.LightningModule):
    def __init__(
        self, embedding_dims: dict[str, ColEmbeddingsParams], h_size, n_classes, lr=1e-3
    ):
        super().__init__()
        self.lstm = TBDTSLstm(
            embedding_dims=embedding_dims, h_size=h_size, n_classes=n_classes
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")

    def training_step(self, batch: PaddedTDTSDBatch, batch_idx):
        # training_step defines the train loop.
        output = self.lstm(batch)
        loss = self.loss(output, batch.labels)
        self.accuracy.update(output, batch.labels)
        self.log("train_acc_step", self.accuracy, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.lr)
        return optimizer
