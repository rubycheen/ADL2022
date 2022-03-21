import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SlotClsDataset
from model import SlotClassifier
from utils import Vocab
from torch.utils.data import DataLoader

import pandas as pd


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SlotClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SlotClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    ).to(device)


    model.eval()

    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device(device)))
    
    # TODO: predict dataset
    preds = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            text = torch.Tensor(vocab.encode_batch([t.split(' ') for t in data['tokens']])).type(torch.LongTensor)
            bz, batch_seq_len = text.shape[0], text.shape[1]
            # seq_str_tags = [t.split(' ') for t in data['tags']]
            text = text.to(device)
            outputs = model(text).view(bz,-1,9)

            for idx, indice in enumerate(data['length']):
                preds += [[dataset.idx2label(p.detach().cpu().item()) for p in outputs[idx,:indice].argmax(dim=1)]]
    
    # TODO: write prediction to file (args.pred_file)
    sep = ' '
    preds = [sep.join(p) for p in preds]
    print('preds', len(preds), preds[0])

    df = pd.DataFrame({'id':[f'test-{i}' for i in range(len(preds))], 'tags': preds})
    df.to_csv(args.pred_file, index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=None)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
