from pathlib import Path
import torch
from torch import Tensor, tensor
from typing import Annotated, Iterable, Hashable, Iterator
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass

# 1D Tensor torch.tensor([1, 3, 7])
SeqTensor = Annotated[Tensor, "1D"]
# sequences (list[Tensor]) – list of variable length sequences, see torch.nn.utils.rnn.pad_sequence
SequenceOfTensors = list[SeqTensor]
# TDTS DataFrame consist of
GroupColumnSequences = dict[Hashable, SeqTensor]
TDTSDataBatch = dict[Hashable, SequenceOfTensors]
# SeqsSampleDict = dict[str, Seqs]
BatchOfTensors = Annotated[Tensor, "2D"]


class TDTSGroupData:
    # TDTSGroupData stores single-group data corresponding to a single data point in the original dataset.
    def __init__(
        self,
        cl_codes_dict: dict[Hashable, list[int]],
        label: int | None = None,
    ):
        self._column_seqs: GroupColumnSequences = {
            col_name: tensor(codes) for col_name, codes in cl_codes_dict.items()
        }
        self._label: int | None = label
        # All column sequences must be the same size.
        self._secs_len = len(self._column_seqs[next(iter(self._column_seqs.keys()))])

    def __getitem__(self, col_name: Hashable) -> SeqTensor:
        return self._column_seqs[col_name]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._column_seqs.keys())

    def __str__(self) -> str:
        return f"tb seqs: {self._column_seqs} \n label: {self.label}"

    def __len__(self):
        return self._secs_len

    @property
    def label(self):
        return self._label


def encode_categorical_cl(
    df_tbts_cl: pd.Series, start=0
) -> tuple[pd.Series, dict[int, str]]:
    codes, uniqs = df_tbts_cl.factorize()
    code2cat_dict: dict[int, str] = dict(enumerate(uniqs, start=start))
    return pd.Series(codes, dtype=int) + start, code2cat_dict


@dataclass(frozen=True)
class ColEmbeddingsParams:
    num_embeddings: int
    embedding_dim: int


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
        df = pd.DataFrame({tbts_groupby_column: df_in[tbts_groupby_column]})
        self._columns_decoder = {}
        self.cat_columns = cat_columns
        for cat_column in cat_columns:
            # The dict starts from 1 because 0 is used for the padded element
            encoded_col, code2cat_dict = encode_categorical_cl(
                df_in[cat_column], start=1
            )
            self._columns_decoder[cat_column] = code2cat_dict
            df[cat_column] = encoded_col
        self._loc_cols_filter = ~df.columns.isin([tbts_groupby_column])
        self._grouped_df_tdts = df.groupby(tbts_groupby_column)
        self._has_labels = False
        if labels_path:
            self._label_cl_name = label_cl_name
            self._has_labels = True
            self._labels: pd.DataFrame = pd.read_parquet(labels_path)
            encoded_col, code2cat_dict = encode_categorical_cl(
                self._labels[label_cl_name]
            )
            self._labels.loc[:, label_cl_name] = encoded_col
            self._labels_decoder = code2cat_dict
        self._ngroups = self._grouped_df_tdts.ngroups

    def __len__(self) -> int:
        return self._ngroups

    def __getitem__(self, idx) -> TDTSGroupData:
        label = (
            self._labels.iloc[idx][self._label_cl_name] if self._has_labels else None
        )
        df_g = self._grouped_df_tdts.get_group(idx).loc[:, self._loc_cols_filter]
        return TDTSGroupData(
            cl_codes_dict=df_g.to_dict(orient="list"),
            label=label,
        )

    @property
    def n_categories(self) -> dict[str, int]:
        return {
            str(column_name): len(code2cat_dict)
            for column_name, code2cat_dict in self._columns_decoder.items()
        }

    def cat_col_embeddings_params(
        self, col_embedding_dims
    ) -> dict[str, ColEmbeddingsParams]:
        return {
            str(column_name): ColEmbeddingsParams(
                # The dict keys start from 1, and 0 is a default value for padded elements, so:
                num_embeddings=len(code2cat_dict) + 1,
                embedding_dim=col_embedding_dims[column_name],
            )
            for column_name, code2cat_dict in self._columns_decoder.items()
        }

    @property
    def n_calsses(self):
        return len(self._labels_decoder)


class PaddedTDTSDBatch:
    def __init__(self, samples_from_tb: list[TDTSGroupData]):
        """
        collates grouped table samples to the batch of tensors:
        samples_from_tb: list[TDTSDataSample] -> self._batch: TDTSDataBatch
        """
        # save lengths befor padding
        self._lengths: list[int] = []
        self._batch: TDTSDataBatch = defaultdict(SequenceOfTensors)
        self._padadded_batch: dict[Hashable, BatchOfTensors] = defaultdict(Tensor)
        labels: list = []
        for sample in samples_from_tb:
            self._lengths.append(len(sample))
            labels.append(sample.label)
            for column in sample:
                s: SeqTensor = sample[column]
                self._batch[column].append(s)
        self._labels: Tensor = tensor(labels)

        for column in self._batch.keys():
            # The default value for padded elements is 0
            self._padadded_batch[column] = torch.nn.utils.rnn.pad_sequence(
                self._batch[column],
                batch_first=True,
            )

    def __getitem__(self, column: Hashable):
        return self._padadded_batch[column]

    def __iter__(self) -> Iterable[Hashable]:
        return iter(self._batch.keys())

    def __len__(self):
        return len(self._batch.keys())

    @property
    def labels(self) -> Tensor:
        return self._labels

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def to(self, device):
        self._padadded_batch = {
            col: t.to(device=device) for col, t in self._padadded_batch.items()
        }
        self._labels = self._labels.to(device=device)


def dummy_collate_fn(samples_from_tb: list[TDTSGroupData]) -> list[TDTSGroupData]:
    return samples_from_tb


def collate_padded_batch_fn(samples_from_tb: list[TDTSGroupData]) -> PaddedTDTSDBatch:
    # batch of padded tensors
    return PaddedTDTSDBatch(samples_from_tb)
