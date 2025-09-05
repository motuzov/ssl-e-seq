import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from seq_loader.names_loader import (
    TDTSDataset,
    collate_padded_batch_fn,
    PaddedTDTSDBatch,
    ColEmbeddingsParams,
    PADDING_VALUE,
)

from seq_loader.nn import CatColumnsDataEncoder


class CharRNN(nn.Module):
    def __init__(
        self, embedding_dims: dict[str, ColEmbeddingsParams], h_size, n_classes
    ):
        super().__init__()
        self.encoder = CatColumnsDataEncoder(embedding_dims)
        self.lstm = nn.LSTM(self.encoder.embedding_dim, h_size, batch_first=True)
        self.h2o = nn.Linear(in_features=2 * h_size, out_features=n_classes)

    def forward(self, batch: PaddedTDTSDBatch):
        encoded_input = self.encoder(batch)
        lstm_h, _ = self.lstm(encoded_input)
        h_n = lstm_h[:, -1]
        avg_pool = torch.mean(lstm_h, 1)
        rnn_out = torch.cat((avg_pool, h_n), 1)
        output = self.h2o(rnn_out)
        return output


def train(
    model,
    tdts_dataset: TDTSDataset,
    n_epoch=10,
    n_batch_size=64,
    report_every=50,
    learning_rate=0.2,
):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loader = torch.utils.data.DataLoader(
        dataset=tdts_dataset,
        shuffle=True,
        batch_size=n_batch_size,
        collate_fn=collate_padded_batch_fn,
    )
    torch.manual_seed(0)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, n_epoch + 1):
        sum_loss = 0
        n_loss = 0
        n_true = 0
        n_predicts = 0
        n_batches = 0
        model.zero_grad()  # clear the gradient
        batch: PaddedTDTSDBatch
        for batch in loader:
            output = model(batch)

            # print(batch.labels)
            # print(output)
            loss = criterion(output, batch.labels)
            with torch.inference_mode():
                sum_loss += criterion(output, batch.labels)
                predictions = output.data.max(dim=1, keepdim=True)[1].view(-1)
                n_predicts += len(predictions)
                n_true += int(sum((predictions == batch.labels)))
                n_loss += 1

            # optimize parameters
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

        if epoch % report_every == 0:
            print(f"epoch: {epoch}")
            print(f"cross_entr={sum_loss / n_loss}")
            print(f"accur={float(n_true) / float(n_predicts)}")
