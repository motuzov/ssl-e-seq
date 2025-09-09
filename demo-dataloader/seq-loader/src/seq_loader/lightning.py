import torch
from seq_loader.names_loader import (
    PaddedTDTSDBatch,
    ColEmbeddingsParams,
)
import lightning as L
from seq_loader.nn import TBDTSLstm
from torchmetrics.classification import MulticlassAccuracy


class LitBaselineLSTM(L.LightningModule):
    def __init__(
        self, embedding_dims: dict[str, ColEmbeddingsParams], h_size, n_classes, lr=1e-3
    ):
        super().__init__()
        self.b_lstm = TBDTSLstm(embedding_dims, h_size, n_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")

    def training_step(self, batch: PaddedTDTSDBatch, batch_idx):
        # training_step defines the train loop.
        output = self.b_lstm(batch)
        loss = self.loss(output, batch.labels)
        acc = self.accuracy(output, batch.labels)
        metrics = {"acc": acc, "cross_entr": loss}
        self.log_dict(
            dictionary=metrics, on_epoch=True, batch_size=len(batch), prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.b_lstm(batch)
        loss = self.loss(output, batch.labels)
        acc = self.accuracy(output, batch.labels)
        metrics = {"val_acc": acc, "val_cross_entr": loss}
        self.log_dict(
            dictionary=metrics, on_epoch=True, batch_size=len(batch), prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def transfer_batch_to_device(
        self, batch, device, dataloader_idx
    ) -> PaddedTDTSDBatch:
        if isinstance(batch, PaddedTDTSDBatch):
            # move all tensors in your custom data structure to the device
            batch.to(device)
        return batch
