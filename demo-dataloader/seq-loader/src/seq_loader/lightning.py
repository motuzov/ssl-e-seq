import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from seq_loader.names_loader import (
    char2idx,
    SeqsDataset,
    TbDataSample,
    dummy_collate_fn,
    collate_padded_batch_fn,
    NAME_CHARS_SEQ,
    PaddedTbBatch,
)


class CharRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0
        )
        self.rnn = nn.LSTM(self.embedding.embedding_dim, hidden_size)
        print(2 * hidden_size)
        self.h2o = nn.Linear(in_features=2 * hidden_size, out_features=n_classes)

    def forward(self, batch):
        input_e = self.embedding(batch)
        lstm_h, _ = self.rnn(input_e)
        h_n = lstm_h[:, -1]
        avg_pool = torch.mean(lstm_h, 1)
        rnn_out = torch.cat((avg_pool, h_n), 1)
        output = self.h2o(rnn_out)
        return output


def train(
    rnn,
    seqs_dataset: SeqsDataset,
    device,
    n_epoch=10,
    n_batch_size=64,
    report_every=50,
    learning_rate=0.2,
):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    loader = torch.utils.data.DataLoader(
        dataset=seqs_dataset,
        shuffle=True,
        batch_size=n_batch_size,
        collate_fn=collate_padded_batch_fn,
    )
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, n_epoch + 1):
        sum_loss = 0
        n_loss = 0
        n_true = 0
        n_predicts = 0
        rnn.zero_grad()  # clear the gradient
        batch: PaddedTbBatch
        for batch in loader:
            first_col: str = next(iter(batch))
            padded_t_batch = batch[first_col]
            padded_t_batch = padded_t_batch.to(device)
            labels = batch.labels.to(device)
            output = rnn(padded_t_batch)
            loss = criterion(output, labels)
            with torch.inference_mode():
                sum_loss += criterion(output, labels)
                predictions = output.data.max(dim=1, keepdim=True)[1].view(-1)
                n_predicts += len(predictions)
                n_true += int(sum((predictions == labels)))
                n_loss += 1

            # optimize parameters
            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

        if epoch % report_every == 0:
            print(f"epoch: {epoch}")
            print(sum_loss / n_loss)
            print(float(n_true) / float(n_predicts))
