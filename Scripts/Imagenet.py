import os
tmp = os.environ.get('SLURM_TMPDIR')
print('node:', os.environ.get('SLURM_NODEID'), ':\n' ,os.listdir(tmp))
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from AdaBelief import AdaBelief
from RAdam import RAdam
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import numpy as np
from ConsciousLR import ConsciousLR
from RAdamConsciousLR import RAdamConsciousLR
import time
from tqdm import tqdm
import json
from pathlib import Path
print('torchvision imported successfully')
import argparse
import warnings
warnings.filterwarnings("ignore")

print('Imports successful!!!')

parser = argparse.ArgumentParser(description='Imagenet classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.0001, type=float, help='')
parser.add_argument('--dataset', default='imagenet', help='cifar10, cifar100, imagenet')
parser.add_argument('--optim', default='Cons', help='Cons, Agg, AdamW')

parser.add_argument('--wd', type=float, default=0.0, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--model_size', type=int, default=18, help='')
parser.add_argument('--max_epochs', type=int, default=120, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--seed', type=int, default=0, help='')

def main():

    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    print(f"Starting...\noptimizer: {args.optim}\nlr: {args.lr}\nbatch size: {args.batch_size}\nseed: {args.seed}\nnum workers: {args.num_workers}, n_gpus: {n_gpus}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f'resnet size: {args.model_size}')
    if int(args.model_size)==18:
        net = torchvision.models.resnet.resnet18(pretrained=False)
    elif int(args.model_size)==34:
        net = torchvision.models.resnet.resnet34(pretrained=False)
    elif int(args.model_size)==50:
        net = torchvision.models.resnet.resnet50(pretrained=False)
    net = torch.nn.DataParallel(net)
    net = net.to('cuda')
    criterion = nn.CrossEntropyLoss()

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    datadir = os.path.join(tmp, 'ILSVRC/Data/CLS-LOC/')
    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'val')

    project = Path(os.environ.get('project'))
    pathLog = project/'logs/imagenet'

    train_transforms = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize
                                ])
    train_dataset = ImageFolder(
                                traindir,
                                train_transforms
                                )
    print(f'train transforms: {train_transforms}')
    valid_dataset = ImageFolder(
                                valdir,
                                transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize
                                ]))
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
                            train_dataset, batch_size=args.batch_size, shuffle=True,
                            worker_init_fn=seed_worker, generator=g,
                            num_workers=args.num_workers, drop_last=False, pin_memory=True)
    valid_loader = DataLoader(
                            valid_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False, pin_memory=True)
    
    stepSize = len(train_loader)
    print(f'stepSize is {stepSize}')

    
    if args.optim=='AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=args.wd)
    elif args.optim=='Cons':
        optimizer = ConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=float(args.wd))
    elif args.optim == 'Agg':
        optimizer = ConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=2., lrLow=.5)
    elif args.optim == 'RAdamCons':
        optimizer = RAdamConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd)
    elif args.optim == 'RAdamAgg':
        optimizer = RAdamConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=2., lrLow=.5)
    elif args.optim == 'RAdam':
        optimizer = RAdam(net.parameters(), lr=float(args.lr), weight_decay=args.wd)
    elif args.optim == 'AdaBelief':
        optimizer = AdaBelief(net.parameters(), lr=float(args.lr), weight_decay=args.wd)
    else:
        print('optimizer not implemented!!!')
    print(optimizer)
    
    top1s=[]
    top5s=[]
    lossList=[]
    start = time.time()
    for epoch in range(args.max_epochs):
        end = time.time()
        losses = train(epoch, net, criterion, optimizer, train_loader)
        losses = losses.item()
        lossList.append(losses)
        top1avg, top5avg = valid(epoch, net, criterion, optimizer, valid_loader)
        top1avg = top1avg.item()
        top5avg = top5avg.item()
        top1s.append(top1avg)
        top5s.append(top5avg)

        log = {
                'train_losses': lossList,
                "top1s": top1s,
                'top5s': top5s,
                'epochs': list(range(epoch+1)),
                }
        with open(pathLog/f'{args.optim}_{args.lr}.json', 'w') as fp:
            json.dump(log, fp)

        print(f'acc1 = {top1avg}, acc5 = {top5avg}, train losses = {losses}, epoch {epoch}, epochTime: {time.time()-end:2.4f}')
    print("total time: ", time.time()-start)

def train(epoch, net, criterion, optimizer, train_loader):
    net.train()

    train_loss = 0
    # correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate((train_loader)):
       
       inputs = inputs.to('cuda', non_blocking=True)
       targets = targets.to('cuda', non_blocking=True)
       outputs = net(inputs)
       loss = criterion(outputs, targets)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()


       train_loss += loss.detach()
       total += targets.size(0)

    return train_loss/total

def valid(epoch, net, criterion, optimizer, valid_loader):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    net.eval()
    # train_loss = 0
    # correct = 0
    # total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((valid_loader)):

            inputs = inputs.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking=True)
            outputs = net(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

        return top1.avg, top5.avg 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__=='__main__':
   main()