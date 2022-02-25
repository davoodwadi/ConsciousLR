import os
import torch
from ConsciousLR import ConsciousLR
from RAdamConsciousLR import RAdamConsciousLR
import numpy as np
from torch import nn
import json
import torch.nn.functional as F
import torch.optim as O
from AdaBelief import AdaBelief
from RAdam import RAdam
from pathlib import Path
import torchtext
from torchtext.data.utils import get_tokenizer
import random
from tqdm import tqdm
import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import WikiText103, WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Transformer Wikitext models, distributed data parallel test')
parser.add_argument('--lr', default=0.0001, type=float, help='')
parser.add_argument('--optim', default='Cons', help='Cons, Agg, AdamW, AdaBelief, RAdam')
parser.add_argument('--dataset', default='WikiText103', help='WikiText103, WikiText2')
parser.add_argument('--emsize', type=int, default=200, help='')
parser.add_argument('--d_hid', type=int, default=200, help='')
parser.add_argument('--nlayers', type=int, default=2, help='')
parser.add_argument('--nhead', type=int, default=2, help='')
parser.add_argument('--dropout', type=float, default=0.2, help='')
parser.add_argument('--wd', type=float, default=0.0, help='')
parser.add_argument('--batch_size', type=int, default=20, help='')
parser.add_argument('--bptt', type=int, default=35, help='')
parser.add_argument('--max_epochs', type=int, default=6, help='')
parser.add_argument('--seed', type=int, default=0, help='')

def main():
    args = parser.parse_args()
    tqdm.monitor_interval = 0
    tmp = os.environ.get('SLURM_TMPDIR')
    scratch = os.environ.get('SCRATCH')
    project = os.environ.get('project')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)

    if args.dataset == 'WikiText103':
        train_iter = WikiText103(root=tmp, split='train')
        print(f'dataset {args.dataset}')
    elif args.dataset == 'WikiText2':
        train_iter = WikiText2(root=tmp, split='train')
        print(f'dataset {args.dataset}')
    else:
        print('dataset not implemented!')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    if args.dataset == 'WikiText103':
        train_iter, val_iter, test_iter = torchtext.datasets.WikiText103(root=tmp, split=('train', 'valid', 'test'))
    elif args.dataset == 'WikiText2':
        train_iter, val_iter, test_iter = torchtext.datasets.WikiText2(root=tmp, split=('train', 'valid', 'test'))
    else:
        print('dataset not implemented!')
    path = Path.cwd()
    if args.dataset == 'WikiText103':
        pathLog = path/'logs/wikitext103'
        pathSaved = path/'saved'
    else:
        pathLog = path/'logs/wikitext2'
        pathSaved = path/'saved/wikitext2'
    def data_process(raw_text_iter):
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def batchify(data, bsz):
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data

    batch_size = args.batch_size
    eval_batch_size = int(args.batch_size//2)
    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    bptt = args.bptt
    def get_batch(source, i):
        """
        Args:
            source: Tensor, shape [full_seq_len, batch_size]
            i: int

        Returns:
            tuple (data, target), where data has shape [seq_len, batch_size] and
            target has shape [seq_len * batch_size]
        """
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    ntokens = len(vocab)  # size of vocabulary
    emsize = args.emsize  # embedding dimension
    d_hid = args.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = args.nhead  # number of heads in nn.MultiheadAttention
    dropout = args.dropout  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    n_gpus = torch.cuda.device_count()
    print(f'batch size: {batch_size}, bptt: {bptt}, seed: {args.seed}, ngpus: {n_gpus}')
    print(f'len vocab: {ntokens}, embeddingSize: {emsize}, hiddenDim: {d_hid}')
    print(f'nlayers: {nlayers}, nAttentionHead: {nhead}, dropout: {dropout}')

    def train(model, train_data, bptt):
        model.train()  # turn on train mode
        total_loss = 0.
        count = 0
        log_interval = 5000
        # start_time = time.time()
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        num_batches = len(range(0, train_data.size(0) - 1, bptt))
        progress = tqdm(total=num_batches)
        for batch, i in enumerate((range(0, train_data.size(0) - 1, bptt))):
            # if batch<140000: continue
            data, targets = get_batch(train_data, i)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = data.size(0)
            if batch_size != bptt:  # only on last batch
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.detach() * batch_size
            count += 1

            if batch%log_interval==0 and batch!=0:
                progress.update(log_interval)

        return total_loss / (len(train_data) - 1), count

    def evaluate(model, eval_data, bptt):
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(bptt).to(device)
        
        with torch.no_grad():
            for batch, i in enumerate(range(0, eval_data.size(0) - 1, bptt)):
                data, targets = get_batch(eval_data, i)
                data = data.to(device)
                targets = targets.to(device)
                batch_size = data.size(0)
                if batch_size != bptt:
                    src_mask = src_mask[:batch_size, :batch_size]
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += batch_size * criterion(output_flat, targets)

        return total_loss / (len(eval_data) - 1)

    stepSize = len(range(0, train_data.size(0) - 1, bptt))
    lr = args.lr
    if args.optim == 'AdamW':
        optimizer = O.AdamW(model.parameters(), lr, weight_decay=args.wd)
    elif args.optim == 'Cons':
        optimizer = ConsciousLR(model.parameters(), stepSize, lr, weight_decay=args.wd)
    elif args.optim == 'Agg':
        optimizer = ConsciousLR(model.parameters(), stepSize, lr, weight_decay=args.wd, lrHigh=2., lrLow=.5)
    elif args.optim == 'RAdamCons':
        optimizer = RAdamConsciousLR(model.parameters(), stepSize, lr, weight_decay=args.wd)
    elif args.optim == 'RAdamAgg':
        optimizer = RAdamConsciousLR(model.parameters(), stepSize, lr, weight_decay=args.wd, lrHigh=2., lrLow=.5)
    elif args.optim == 'RAdam':
        optimizer = RAdam(model.parameters(), lr, weight_decay=args.wd)
    elif args.optim == 'AdaBelief':
        optimizer = AdaBelief(model.parameters(), lr, weight_decay=args.wd)
    else:
        print('optimizer not implemented!!!')
    print(optimizer)

    best_test_loss = float('inf')
    epochs = args.max_epochs
    train_losses = []
    val_ppls = []
    test_ppls = []
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        trainLoss, count = train(model, train_data, bptt)
        trainLoss = trainLoss.item()
        train_losses.append(trainLoss)

        val_loss = evaluate(model, val_data, bptt)
        val_loss = val_loss.item()
        val_ppl = math.exp(val_loss)
        val_ppls.append(val_ppl)

        test_loss = evaluate(model, test_data, bptt)
        test_loss = test_loss.item()
        test_ppl = math.exp(test_loss)
        test_ppls.append(test_ppl)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | trainLoss: {trainLoss:5.2f}'
            f' | valid ppl {val_ppl:8.2f}| test ppl {test_ppl:8.2f} |')
        print('-' * 89)

        if test_loss < best_test_loss:
            dic = {
                'model': model,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'train_loss': trainLoss,
                'test_ppl': test_ppl
            }
            if args.dataset == 'WikiText103':
                torch.save(dic, pathSaved/f'{args.optim}_{args.lr}_103model.pt')
            else:
                torch.save(dic, pathSaved/f'{args.optim}_{args.lr}_2model.pt')
            
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_train_loss = trainLoss
            best_test_loss = test_loss
            best_test_ppl = test_ppl
        log = {
                'train_losses': train_losses,
                "val_ppls": val_ppls,
                'test_ppls': test_ppls,
                'best_epoch': best_epoch,
                'best_val_ppl': best_val_ppl,
                'best_val_loss': best_val_loss,
                'best_train_loss': best_train_loss,
                'best_test_ppl': best_test_ppl,
                'best_test_loss': best_test_loss
                }
        if args.dataset == 'WikiText103':
            with open(pathLog/f'{args.optim}_{args.lr}_103.json', 'w') as fp:
                json.dump(log, fp)
        else:
            with open(pathLog/f'{args.optim}_{args.lr}_2.json', 'w') as fp:
                json.dump(log, fp)

    print(f'best test ppl: {best_test_ppl}')
    print(log)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

if __name__=='__main__':
   main()