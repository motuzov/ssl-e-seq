from seq_loader.names_loader import (
    TDTSDataset,
    TDTSGroupData,
    dummy_collate_fn,
    collate_padded_batch_fn,
    PaddedTDTSDBatch,
    SeqTensor,
)
import pytest
from pathlib import Path
from dotenv import load_dotenv
import os


import torch
from torch import tensor
from torchmetrics import Accuracy


load_dotenv()


_NAME_CHARS_COL = "name_chars"


@pytest.fixture
def tdts_dataset() -> TDTSDataset:
    data_path = Path(str(os.getenv("DATA_DIR")))
    filename = "tbts-names"
    return TDTSDataset(
        tbts_path=data_path / f"{filename}.parquet",
        tbts_groupby_column="G",
        cat_columns=[_NAME_CHARS_COL],
        labels_path=data_path / f"{filename}-labels.parquet",
    )


def test_TDTSDataset_init(tdts_dataset: TDTSDataset):
    assert tdts_dataset._columns_decoder["name_chars"][1] == "K"
    ncodes = len(tdts_dataset._columns_decoder["name_chars"])
    assert ncodes == 87
    assert tdts_dataset._columns_decoder["name_chars"][87] == "ñ"


def test_dataset_getitm(tdts_dataset: TDTSDataset):
    # get the sample(the datapoint ) by idx generated with DataLoader's sampler
    first_group_data: TDTSGroupData = tdts_dataset[0]
    seq: SeqTensor = first_group_data[_NAME_CHARS_COL]
    assert torch.equal(tensor([1, 2, 3, 4, 5, 6]), seq)
    # assert 0 == datapoint.label


def test_compatibility_with_dataloader(tdts_dataset: TDTSDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=tdts_dataset, shuffle=True, batch_size=4, collate_fn=dummy_collate_fn
    )
    first_batch_of_sampels: list[TDTSGroupData] = next(iter(loader))
    first_sample: TDTSGroupData = first_batch_of_sampels[0]
    with capsys.disabled():
        print(f"\nfirst data point in batch:\n{first_sample}\n")
    s = first_sample[_NAME_CHARS_COL]
    assert torch.equal(torch.tensor([24, 8, 32, 8, 25, 15, 25]), s)
    assert first_sample.label == 15
    # assert len(first_sample) == 1


def test_collate_padded_batch_fn(tdts_dataset: TDTSDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=tdts_dataset,
        shuffle=True,
        batch_size=4,
        collate_fn=collate_padded_batch_fn,
    )
    first_batch: PaddedTDTSDBatch = next(iter(loader))
    first_col_name = next(iter(first_batch))
    with capsys.disabled():
        print(f"Padded batch of cl={first_col_name}: \n{first_batch[first_col_name]}\n")
        print(f"len(columns num) : {len(first_batch)}")
        print(f"Labels: {first_batch.labels}")
    assert True


def test_loader_with_rnn(tdts_dataset: TDTSDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=tdts_dataset,
        shuffle=True,
        batch_size=4,
        collate_fn=collate_padded_batch_fn,
    )
    first_batch: PaddedTDTSDBatch = next(iter(loader))
    first_col: str = next(iter(first_batch))
    batch = first_batch[first_col]
    embedding = torch.nn.Embedding(
        num_embeddings=tdts_dataset.num_embeddings(_NAME_CHARS_COL),
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
        in_features=2 * h_size, out_features=tdts_dataset.n_calsses
    )
    l_out = linear_layer(out)
    loss_layer = torch.nn.CrossEntropyLoss()
    loss = loss_layer(l_out, first_batch.labels)
    with capsys.disabled():
        print(l_out.shape)
        print(tdts_dataset.n_calsses)
        print(first_batch.labels.shape)
        print(loss)
    with capsys.disabled():
        print("accuracy")
        print(l_out)
        predictions = l_out.data.max(dim=1, keepdim=True)[1].view(-1)
        print(predictions)
        print(first_batch.labels)
        accuracy = Accuracy(task="multiclass", num_classes=tdts_dataset.n_calsses)
        acc = accuracy(predictions, first_batch.labels)
        print(predictions == first_batch.labels)
        print(acc)
        print(sum(torch.tensor([1, 1, 1]) == torch.tensor([1, 0, 1])))
