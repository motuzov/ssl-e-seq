import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from seq_loader.names_loader import (
    ColEmbeddingsParams,
    PaddedTDTSDBatch,
    PADDING_VALUE,
)


class CatColumnsDataEncoder(nn.Module):
    def __init__(self, embedding_dims: dict[str, ColEmbeddingsParams]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(
                    num_embeddings=col_embedings_params.num_embeddings,
                    embedding_dim=col_embedings_params.embedding_dim,
                    padding_idx=PADDING_VALUE,
                )
                for col, col_embedings_params in embedding_dims.items()
            }
        )

    def forward(self, padded_batch: PaddedTDTSDBatch):
        return torch.cat(
            [
                self.embeddings[col_name](padded_batch[col_name])
                for col_name in self.embeddings.keys()
            ]
        )

    @property
    def embedding_dim(self):
        return sum([e.embedding_dim for e in self.embeddings.values()])
