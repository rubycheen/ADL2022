import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tarfile import LENGTH_LINK
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from torch.utils.data import DataLoader

from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }


    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False)
    

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_rate=args.dropout, bidirectional=args.bidirectional, num_class=150, attn=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    model.to(device)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=5e-5, verbose=True)

    best_acc = 0.9330

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_total_acc, train_total_count = 0, 0
        for idx, (data) in enumerate(train_loader):
            model.zero_grad()
            text = torch.Tensor(vocab.encode_batch([t.split(' ') for t in data['text']])).type(torch.LongTensor)    
            text, intent = text.to(device), data['intent'].to(device)
            outputs = model(text)
            loss = criterion(outputs, intent)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            train_total_acc += (preds == intent).sum().item()
            train_total_count += intent.size(0)
        print('\n\t Training Accuracy: \t {:.2f}%'.format(100*(train_total_acc/train_total_count)))
    
        model.eval()
        val_total_acc, val_total_count = 0, 0
        for idx, (data) in enumerate(valid_loader):
            model.zero_grad()
            text = torch.Tensor(vocab.encode_batch([t.split(' ') for t in data['text']])).type(torch.LongTensor)    
            text, intent = text.to(device), data['intent'].to(device)
            outputs = model(text)
            loss = criterion(outputs, intent)
            preds = outputs.argmax(dim=1)
            val_total_acc += (preds == intent).sum().item()
            val_total_count += intent.size(0)
        print('\n\t Valid Accuracy: \t {:.2f}%'.format(100*(val_total_acc/val_total_count)))
        if val_total_acc/val_total_count > best_acc:
            best_acc = val_total_acc/val_total_count
            torch.save(model.state_dict(), './models/best.pt')
            print('\n\t Best Accuracy! Model Saved.\n')

        scheduler.step(loss)

        if val_total_acc/val_total_count>0.86:
            torch.save(model.state_dict(), f'./models/epoch-{epoch}.pt')
            print('\n\t Model Saved.\n')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=None)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

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
