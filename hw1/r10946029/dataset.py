from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:

        data = {}

        # data['index'] = index
        data['text'] = self.data[index]['text']
        data['intent'] = self.label2idx(self.data[index]['intent']) if 'intent' in self.data[index].keys() else -1

        return data
    
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SlotClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        data = {}
        data['length'] = len(self.data[index]['tokens'])
        data['tokens'] = self.data[index]['tokens']
        data['tags'] = [str(self.label2idx(t)) for t in self.data[index]['tags']] if 'tags' in self.data[index].keys() else -1

        sep = " "
        data['tokens'] =  sep.join(data['tokens'])
        data['tags'] =  sep.join( data['tags'] ) if 'tags' in self.data[index].keys() else -1

        return data
    
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
