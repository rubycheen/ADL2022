import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tarfile import LENGTH_LINK
from typing import Dict
import numpy as np

import torch
from tqdm import trange

from dataset import SlotClsDataset
from utils import Vocab
from torch.utils.data import DataLoader

from model import SlotClassifier

from seqeval.scheme import IOB2
from seqeval.metrics import accuracy_score, classification_report

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SlotClsDataset] = {
        split: SlotClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }


    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False)
    

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SlotClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_rate=args.dropout, bidirectional=args.bidirectional, num_class=9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Using {device}.')
    model.to(device)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=5e-5, verbose=True)

    best_acc = 0.9671

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        preds, target = [], []
        for idx, data in enumerate(train_loader):
            model.zero_grad()
            text = torch.Tensor(vocab.encode_batch([t.split(' ') for t in data['tokens']])).type(torch.LongTensor)
            bz = text.shape[0]
            batch_seq_len = text.shape[1]
            seq_str_tags = [t.split(' ') for t in data['tags']]
            tags = []
            for line in seq_str_tags:
                pad_line = []
                for tag in line:
                    pad_line.append(int(tag))
                while len(pad_line) < batch_seq_len:
                    pad_line.append(-1)
                tags.append(pad_line)
            tags = torch.LongTensor(tags)
            text, tags = text.to(device), tags.to(device)
            outputs = model(text)
            tags=tags.view(-1)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            # print('text.shape[1]', text.shape[1])
            tags = tags.view(bz,-1) #[batch, seq_len]
            # print('*tags.shape',tags.shape)
            outputs = outputs.view(bz,-1,9) #[batch, seq_len, cls_probs]
            for idx, indice in enumerate(data['length']):
                # preds += [outputs[idx,:indice].argmax(dim=1).tolist()]
                preds += [[datasets[TRAIN].idx2label(p.detach().cpu().item()) for p in outputs[idx,:indice].argmax(dim=1)]]
                # target += [tags[idx,:indice].tolist()]
                target += [[datasets[TRAIN].idx2label(p.detach().cpu().item()) for p in tags[idx,:indice]]]

        print('\n\t Training Accuracy: \t {:.2f}%'.format(accuracy_score(preds, target)*100))
    
        model.eval()
        preds, target = [], []
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):
                # model.zero_grad()
                text = torch.Tensor(vocab.encode_batch([t.split(' ') for t in data['tokens']])).type(torch.LongTensor)
                bz, batch_seq_len = text.shape[0], text.shape[1]
                seq_str_tags = [t.split(' ') for t in data['tags']]
                tags = []
                for line in seq_str_tags:
                    pad_line = []
                    for tag in line:
                        pad_line.append(int(tag))
                    while len(pad_line) < batch_seq_len:
                        pad_line.append(-1)
                    tags.append(pad_line)
                tags = torch.LongTensor(tags)
                text, tags = text.to(device), tags.to(device)
                outputs = model(text)
                tags = tags.view(-1)
                loss = criterion(outputs, tags)

                tags = tags.view(bz,-1)
                # print('*tags.shape',tags.shape)
                outputs = outputs.view(bz,-1,9)

                # print(data['length'].shape)
                for idx, indice in enumerate(data['length']):
                    # preds += [outputs[idx,:indice].argmax(dim=1).tolist()]
                    preds += [[datasets[TRAIN].idx2label(p.detach().cpu().item()) for p in outputs[idx,:indice].argmax(dim=1)]]
                    # target += [tags[idx,:indice].tolist()]
                    target += [[datasets[TRAIN].idx2label(p.detach().cpu().item()) for p in tags[idx,:indice]]]
        acc = accuracy_score(preds, target)
        print('\n\t Valid Accuracy: \t {:.2f}%'.format(acc*100))


        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './models/slot-best.pt')
            print('\n\t Best Accuracy! Model Saved.\n')


        scheduler.step(loss)

        if acc>0.965:
            torch.save(model.state_dict(), f'./models/slot-epoch-{epoch}.pt')
            print('\n\t Model Saved.\n')
    
        if epoch % 10 == 9:
            print(classification_report(preds, target, mode='strict', scheme=IOB2))



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=None)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
