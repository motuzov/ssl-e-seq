import string
from pathlib import Path
import torch
from torch import Tensor
from typing import Annotated, Any
from collections import defaultdict
import pandas as pd


# Seqs is a list of 1d tensors various lenghts [torch.Tensor(1, 4, 3), torch.Tensor(2, 1)]
# 1D Tensor torch.tensor([1, 3, 7])
# seq: Seq, so seq.dim must be 1
Seq = Annotated[Tensor, "1D"]
# sequences (list[Tensor]) – list of variable length sequences, see torch.nn.utils.rnn.pad_sequence
Seqs = list[Seq]
# sequence consists of group's values of one column
# sequences of the one sample
SeqDict = dict[str, Seq]
TbBatchSeqs = dict[str, Seqs]
# SeqsSampleDict = dict[str, Seqs]
Texts = list[str]


# data point data
# dict of
#  L is length of the sequence
# where B is the batch size, L is length of the sequence, T is the length of the longest sequence in the batch, *
# grouped_df.get_group(gidx).loc[: , df.columns != 'g'] in the Seqs Table group releteded to data point data


class TDTSDataSample:
    # Tabular data point represent the table's data for the one group
    # object this type reterning by Dataset __getitem__
    # the type represent sequential data of one data point selected from the grouped table
    def __init__(self, seqs_dict: list[int], label: int | None = None):
        self._seqs: SeqDict = {col_name: codes for col_name, codes in seqs_dict.items()}
        self._label: int | None = label
        self._secs_len = len(self._seqs[next(iter(self._seqs.keys()))])

    def __getitem__(self, col_name: str) -> Seq:
        # Seq, label, description
        # TODO: Specify the type of the SeqData
        return self._seqs[col_name]

    def __iter__(self):
        return iter(self._seqs.keys())

    def __str__(self) -> str:
        return f"label: {self._label} \n tb seqs: {self._seqs}"

    def __len__(self):
        return self._secs_len

    @property
    def label(self):
        return self._label


def encode_categorical_cl(
    df_tbts_cl: pd.Series,
) -> tuple[pd.Series, dict[str, dict[str, int]]]:
    codes, uniqs = df_tbts_cl.factorize()
    # 0 reserved as the padded idx
    cod2cat = dict(enumerate(uniqs, start=1))
    cat2cod = {cat: idx for idx, cat in cod2cat.items()}
    d = {
        "cat2cod": cat2cod,
        "cod2cat": cod2cat,
    }
    return pd.Series(codes, dtype=int) + 1, d


class TDTSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tbts_path: Path,
        tbts_groupby_column: str,
        cat_columns: list[str] = [],
        num_columns: list[str] = [],
        labels_path: Path | None = None,
        label_cl_name="labels",
    ) -> None:
        self._tbts_path = tbts_path
        self._tbts_groupby_column = tbts_groupby_column
        df_in = pd.read_parquet(tbts_path, engine="pyarrow")
        _df = pd.DataFrame({tbts_groupby_column: df_in[tbts_groupby_column]})
        self._encode_decode_dicts = {}
        for cat_column in cat_columns:
            encoded_col, d = encode_categorical_cl(df_in[cat_column])
            self._encode_decode_dicts[cat_column] = d
            _df[cat_column] = encoded_col
        self._grouped_df_tdts = _df.groupby(tbts_groupby_column)
        self._has_labels = False
        if labels_path:
            self._labels: pd.DataFrame = pd.read_parquet(labels_path)
            encoded_col, d = encode_categorical_cl(self._labels[label_cl_name])
            self._labels.loc[:, label_cl_name] = encoded_col
            self._encode_decode_label_dicts = d
        self._ngroups = self._grouped_df_tdts.ngroups

    def __len__(self) -> int:
        return self._ngroups

    def __getitem__(self, idx) -> TDTSDataSample:
        label = self._labels.iloc[idx] if self._has_labels else None
        return TDTSDataSample(
            seqs_dict=self._grouped_df_tdts.get_group(idx).to_dict(orient="list"),
            label=label,
        )


class PaddedTbBatch:
    def __init__(self, samples_from_tb: list[TDTSDataSample]):
        """
        samples of the dataset table collate to the batch:
        samples_from_tb: list[TDTSDataSample] -> self._batch: TbBatchSeqs = dict[str, Seqs]
        the batch key is the column name of the dataset tabel
        batch value are the list of sequences
        each sequence represent sequential data of one data point stored in the dataset table
        and we save the original lengths of the sequences into self._lengths before calling pad_sequence
        """
        # save lengths befor padding
        self._lengths: list[int] = []
        self._batch: TbBatchSeqs = defaultdict(Seqs)
        self._padadded_batch: dict[str, Tensor] = defaultdict(Tensor)
        self._labels: list = []
        for sample in samples_from_tb:
            self._lengths.append(len(sample))
            self._labels.append(sample.label)
            for column in sample:
                s = sample[column]
                self._batch[column].append(s)

        for column in self._batch.keys():
            self._padadded_batch[column] = torch.nn.utils.rnn.pad_sequence(
                self._batch[column], batch_first=True
            )

    def __getitem__(self, column):
        return self._padadded_batch[column]

    def __iter__(self) -> int:
        return iter(self._batch.keys())

    def __len__(self):
        return len(self._batch.keys())

    @property
    def labels(self) -> Tensor:
        return torch.tensor(self._labels)

    @property
    def lengths(self) -> list[int]:
        return self._lengths


def dummy_collate_fn(samples_from_tb: list[TDTSDataSample]) -> list[TDTSDataSample]:
    return samples_from_tb


def collate_padded_batch_fn(samples_from_tb: list[TDTSDataSample]) -> PaddedTbBatch:
    # batch of padded tensors
    return PaddedTbBatch(samples_from_tb)
