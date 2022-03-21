from typing import Dict

import torch
from torch.nn import Embedding

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.dim_embeddings = 300
        self.gru = torch.nn.GRU(self.dim_embeddings, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)      
        # self.lstm = torch.nn.LSTM(self.dim_embeddings, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, bias = False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden_size*2, num_class) if bidirectional else torch.nn.Linear(hidden_size, num_class)

    @property
    def encoder_output_size(self) -> int:
        # calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # print('batch', batch.shape)
        # print(batch)
        context = self.embed(batch)
        # context_outs, (context_h_n, _) = self.lstm(context)
        _, context_h_n = self.gru(context)
        # print('context_outs.shape', context_outs.shape)
        context_h_n = self.dropout(context_h_n)
        out = torch.cat((context_h_n[-1], context_h_n[-2]), axis=-1) if self.bidirectional else context_h_n[-1]
        # print('context_h_n.shape', context_h_n.shape)
        out = self.classifier(out)
        return out


class SlotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SlotClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.dim_embeddings = 300
        # self.lstm = torch.nn.LSTM(self.dim_embeddings, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.gru = torch.nn.GRU(self.dim_embeddings, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)      
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden_size*2, num_class) if bidirectional else torch.nn.Linear(hidden_size, num_class)

    @property
    def encoder_output_size(self) -> int:
        # calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # print('batch', batch.shape)
        # print(batch)
        context = self.embed(batch)
        # context_outs, _ = self.lstm(context)
        context_outs, _ = self.gru(context)
        # print('context_outs.shape', context_outs.shape)
        out = self.dropout(context_outs)
        # print('out.shape', out.shape)
        out = self.classifier(out)
        out = out.view(-1,self.num_class)
        # print(out.shape)
        return out
