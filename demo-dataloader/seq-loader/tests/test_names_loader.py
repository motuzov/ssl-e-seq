from seq_loader.names_loader import (
    char2idx,
    SeqsDataset,
    TbDataSample,
    dummy_collate_fn,
    collate_padded_batch_fn,
    NAME_CHARS_SEQ,
    PaddedTbBatch,
)
import pytest
from pathlib import Path
import torch
from torchmetrics import Accuracy


def test_cahr2idx():
    text = "ab_c"
    seq_idxs = [char2idx(c) for c in text]
    assert [0, 1, 57, 2] == seq_idxs


@pytest.fixture
def seqs_dataset():
    return SeqsDataset(
        names_data_path=Path("/home/jovyan/work/data/rnn/data-loader/data/names")
    )


def test_seqsdataset_init(seqs_dataset: SeqsDataset):
    seqs_dataset._file_names
    assert seqs_dataset._file_names == [
        "Arabic.txt",
        "Chinese.txt",
        "Czech.txt",
        "Dutch.txt",
        "English.txt",
        "French.txt",
        "German.txt",
        "Greek.txt",
        "Irish.txt",
        "Italian.txt",
        "Japanese.txt",
        "Korean.txt",
        "Polish.txt",
        "Portuguese.txt",
        "Russian.txt",
        "Scottish.txt",
        "Spanish.txt",
        "Vietnamese.txt",
    ]


def test_dataset_getitm(seqs_dataset: SeqsDataset):
    # get the sample(the datapoint ) by idx generated with DataLoader's sampler
    datapoint: TbDataSample = seqs_dataset[0]
    seq = datapoint[NAME_CHARS_SEQ]
    assert torch.equal(torch.tensor([36, 7, 14, 20, 17, 24]), seq)
    assert 0 == datapoint.label
    assert "Arabic.txt" == datapoint.description


def test_compatibility_with_dataloader(seqs_dataset: SeqsDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=seqs_dataset, shuffle=True, batch_size=4, collate_fn=dummy_collate_fn
    )
    first_batch_of_sampels: list[TbDataSample] = next(iter(loader))
    first_sample: TbDataSample = first_batch_of_sampels[0]
    with capsys.disabled():
        print(f"\nfirst data point in batch:\n{first_sample}\n")
    s = first_sample[NAME_CHARS_SEQ]
    assert torch.equal(torch.tensor([45, 0, 12, 0, 13, 8, 13]), s)
    assert 14 == first_sample.label
    assert "Russian.txt" == first_sample.description


def test_collate_padded_batch_fn(seqs_dataset: SeqsDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=seqs_dataset,
        shuffle=True,
        batch_size=4,
        collate_fn=collate_padded_batch_fn,
    )
    first_batch: PaddedTbBatch = next(iter(loader))
    first_col: str = next(iter(first_batch))
    assert first_batch.lengths == [7, 4, 9, 6]
    assert first_batch.labels == [14, 4, 8, 14]
    assert torch.equal(
        first_batch[first_col],
        torch.tensor(
            [
                [45, 0, 12, 0, 13, 8, 13, 0, 0],
                [43, 14, 19, 7, 0, 0, 0, 0, 0],
                [40, 56, 38, 0, 7, 14, 13, 4, 24],
                [38, 11, 14, 3, 8, 10, 0, 0, 0],
            ]
        ),
    )


def test_loader_with_rnn(seqs_dataset: SeqsDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=seqs_dataset,
        shuffle=True,
        batch_size=4,
        collate_fn=collate_padded_batch_fn,
    )
    first_batch: PaddedTbBatch = next(iter(loader))
    first_col: str = next(iter(first_batch))
    batch = first_batch[first_col]
    embedding = torch.nn.Embedding(
        num_embeddings=seqs_dataset.num_embeddings,
        embedding_dim=4,
        padding_idx=0,
    )
    input_e = embedding(batch)
    h_size = 8
    lstm_layer = torch.nn.LSTM(
        input_size=embedding.embedding_dim,
        hidden_size=h_size,
        batch_first=True,
    )
    lstm_h, _ = lstm_layer(input_e)
    avg_pool = torch.mean(lstm_h, 1)
    h_n = lstm_h[:, -1]
    with capsys.disabled():
        print(batch)
        print(input_e)
        print("LSTM output B x L x H:")
        print(lstm_h.shape)
        # print(lstm_h)
        print("Avr pool  B x H")
        print(avg_pool.shape)
        print(avg_pool)
        print("H_n: B x H")
        print(h_n.shape)
        print(h_n)
        print(lstm_h)

    out = torch.cat((avg_pool, h_n), 1)
    with capsys.disabled():
        print(out)
        print("Out: B x (H + H)")
        print(out.shape)
    linear_layer = torch.nn.Linear(
        in_features=2 * h_size, out_features=seqs_dataset.n_calsses
    )
    l_out = linear_layer(out)
    loss_layer = torch.nn.CrossEntropyLoss()
    loss = loss_layer(l_out, first_batch.labels)
    with capsys.disabled():
        print(l_out.shape)
        print(seqs_dataset.n_calsses)
        print(first_batch.labels.shape)
        print(loss)
    with capsys.disabled():
        print("accuracy")
        print(l_out)
        predictions = l_out.data.max(dim=1, keepdim=True)[1].view(-1)
        print(predictions)
        print(first_batch.labels)
        accuracy = Accuracy(task="multiclass", num_classes=seqs_dataset.n_calsses)
        acc = accuracy(predictions, first_batch.labels)
        print(predictions == first_batch.labels)
        print(acc)
        print(sum(torch.tensor([1, 1, 1]) == torch.tensor([1, 0, 1])))
