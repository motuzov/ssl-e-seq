import string
from pathlib import Path
import unicodedata
import torch
from torch import Tensor
from typing import Annotated, Any
from collections import defaultdict


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


# used as an adapter for TBDataPoint
NAME_CHARS_SEQ = "name_chars_seq"


class TxTbColumnValidaionException(Exception):
    def __init__(self, col_name, message=""):
        self.message = message
        self.col_name = col_name
        super().__init__(self.message)

    def __str__(self):
        return f"SeqTb exception: {self.message}"


class TbDataSample:
    # Tabular data point represent the table's data for the one group
    # object this type reterning by Dataset __getitem__
    # the type represent sequential data of one data point selected from the grouped table
    def __init__(self, seqs_dict: SeqDict, label: Any = None, description: Any = None):
        self._seqs: SeqDict = seqs_dict
        self._label: Any = label
        self._description: Any = description

    def __getitem__(self, seq_name: str) -> Seq:
        # Seq, label, description
        # TODO: Specify the type of the SeqData
        return self._seqs[seq_name]

    def __iter__(self):
        return iter(self._seqs.keys())

    def __str__(self) -> str:
        return f"label: {self._label} \n label description: {self._description} \n seqs: {self._seqs}"

    def __len__(self):
        return len(self._seqs[next(iter(self._seqs.keys()))])

    @property
    def label(self):
        return self._label

    @property
    def description(self):
        return self._description

    # def validate_seq_lengths(gtb_dict: GroupedTbDict):
    #     for col_name, seqs in self._seq_tb.items():
    #         if not self._lengths == list(map(len, seqs)):
    #             raise TxTbColumnValidaionException(
    #                 col_name=col_name,
    #                 message="sequence lengths of the group must be the same, but the length of sequence {seqs} in the column {col_name} is distinct from others.",
    #             )


_allowed_characters = string.ascii_letters + " .,;'" + "_"


def char2idx(c: str) -> int:
    if c not in _allowed_characters:
        c = "_"
    return _allowed_characters.find(c)


def line_to_seq(line: str) -> torch.Tensor:
    return torch.tensor([char2idx(c) for c in line])


def load_labeled_names(
    names_data_path: Path, file_names: list[str]
) -> tuple[Seqs, list[str], Texts]:
    labels: list[str] = []
    seqs: Seqs = []
    texts: Texts = []
    for fnanme in file_names:
        with open(names_data_path / fnanme, mode="r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seqs.append(line_to_seq(line))
                labels.append(fnanme)
                texts.append(line)
    return seqs, labels, texts


def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in _allowed_characters
    )


class SeqsDataset(torch.utils.data.Dataset):
    def __init__(self, names_data_path: Path) -> None:
        self._names_data_path = names_data_path
        self._file_names = sorted(
            [f_path.name for f_path in names_data_path.glob("*.txt")]
        )
        self._l2idx = {name: idx for idx, name in enumerate(self._file_names)}
        self._seqs, self._labels, self._texts = load_labeled_names(
            names_data_path, self._file_names
        )

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx) -> TbDataSample:
        label = self._l2idx[self._labels[idx]]
        seqs_of_one_sample: SeqDict = {NAME_CHARS_SEQ: self._seqs[idx]}
        return TbDataSample(
            seqs_dict=seqs_of_one_sample,
            label=label,
            description=self._labels[idx],
        )

    @property
    def num_embeddings(self) -> int:
        return len(_allowed_characters)

    @property
    def n_calsses(self) -> int:
        return len(self._l2idx)


class PaddedTbBatch:
    def __init__(self, samples_from_tb: list[TbDataSample]):
        """
        samples of the dataset table collate to the batch:
        samples_from_tb: list[TbDataSample] -> self._batch: TbBatchSeqs = dict[str, Seqs]
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


def dummy_collate_fn(samples_from_tb: list[TbDataSample]) -> list[TbDataSample]:
    return samples_from_tb


def collate_padded_batch_fn(samples_from_tb: list[TbDataSample]) -> PaddedTbBatch:
    # batch of padded tensors
    return PaddedTbBatch(samples_from_tb)
