import os
cwd = os.getcwd()
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import random
import json
from ConsciousLR import ConsciousLR
from RAdamConsciousLR import RAdamConsciousLR
from RAdam import RAdam
from AdaBelief import AdaBelief
from ActiveBelief import ActiveBelief
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from pathlib import Path

parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.0001, type=float, help='')
parser.add_argument('--dataset', default='cifar10', help='cifar10, cifar100, imagenet')
parser.add_argument('--optim', default='ActiveAdamW', help='ActiveRAdam, Agg, AdamW, RAdam, AdaBelief, ActiveBelief')
# parser.add_argument('--wdtype', default='W', help='W, G, I')
parser.add_argument('--wd', type=float, default=0.0000, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_epochs', type=int, default=100, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--flip', type=int, default=1, help='True=1, False=0')
parser.add_argument('--crop', type=int, default=1, help='True=1, False=0')


def main():

    args = parser.parse_args()
    print(f"dataset: {args.dataset}\noptimizer: {args.optim}\n")
    print(f"wd: {args.wd}")
    print(f"batch size: {args.batch_size}\nseed: {args.seed}\nnum workers: {args.num_workers}")

    project = Path(os.environ.get('project'))
    pathLog = project/'logs/cifar10'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.dataset == 'imagenet':
        net = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,
                                      [2, 2, 2, 2],
                                      num_classes=1000)
    elif args.dataset == 'cifar10':                                  
        net = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,
                                        [2, 2, 2, 2],
                                        num_classes=10)
    else:
        net = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,
                                        [2, 2, 2, 2],
                                        num_classes=100)

    # net = torch.nn.DataParallel(net)
    net = net.to('cuda')
    print(args.flip)
    if int(args.crop)==1:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif int(args.flip)==1:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif int(args.flip)==0:
        transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_val = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    print(transform_train)

    tmpdir = os.environ.get('SLURM_TMPDIR')
    if args.dataset == 'cifar10':
        dataset_train = CIFAR10(root=tmpdir+'/data', train=True, download=False, transform=transform_train)
        dataset_valid = CIFAR10(root=tmpdir+'/data', train=False, download=False, transform=transform_val)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(args.seed)
        train_loader = torch.utils.data.DataLoader(
                                dataset_train, batch_size=args.batch_size, shuffle=True,
                                worker_init_fn=seed_worker, generator=g,
                                num_workers=args.num_workers, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(
                                dataset_valid, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False)
        
    elif args.dataset == 'cifar100':
        dataset_train = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        dataset_valid = CIFAR100(root='./data', train=False, download=True, transform=transform_val)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,shuffle=False,
                                                    num_workers=args.num_workers, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size,shuffle=False,
                                                    num_workers=args.num_workers, drop_last=False)
    else:
        datadir = os.path.join('~/scratch/', '/ILSVRC/Data/CLS-LOC/')
        traindir = os.path.join(datadir, 'train')
        valdir = os.path.join(datadir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True)

        valid_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False)

    

    stepSize = len(train_loader)
    
    print(f'stepSize is {stepSize}')

    criterion = nn.CrossEntropyLoss()

    if args.optim=='AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=args.wd)
    elif args.optim=='Cons':
        optimizer = ConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=float(args.wd))
    elif args.optim == 'ActiveAdamW':
        optimizer = ConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=2., lrLow=.5)
    elif 'ActiveAdam-' in args.optim :
        lrHigh = float(args.optim.split('-')[1])
        lrLow = float(args.optim.split('-')[2])
        optimizer = ConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=lrHigh, lrLow=lrLow)
    elif 'ActiveBelief-' in args.optim :
        lrHigh = float(args.optim.split('-')[1])
        lrLow = float(args.optim.split('-')[2])
        optimizer = ActiveBelief(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=lrHigh, lrLow=lrLow)
    elif 'ActiveRAdam-' in args.optim :
        lrHigh = float(args.optim.split('-')[1])
        lrLow = float(args.optim.split('-')[2])
        optimizer = RAdamConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=lrHigh, lrLow=lrLow)
    elif args.optim == 'ActiveAdam0505':
        optimizer = ConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=.5, lrLow=.5)
    elif args.optim == 'RAdamCons':
        optimizer = RAdamConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd)
    elif args.optim == 'ActiveRAdam':
        optimizer = RAdamConsciousLR(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=2., lrLow=.5)
    elif args.optim == 'RAdam':
        optimizer = RAdam(net.parameters(), lr=float(args.lr), weight_decay=args.wd)
    elif args.optim == 'AdaBelief':
        optimizer = AdaBelief(net.parameters(), lr=float(args.lr), weight_decay=args.wd)
    elif args.optim == 'ActiveBelief':
        optimizer = ActiveBelief(net.parameters(), stepSize, lr=float(args.lr), weight_decay=args.wd, lrHigh=2., lrLow=.5)    
    else:
        print('optimizer not implemented!!!')
    print(optimizer)
      

    start = time.time()
    accs=[]
    loss_list=[]
    for epoch in range(args.max_epochs):
        losses = train(epoch, net, criterion, optimizer, train_loader)
        losses = losses.item()
        loss_list.append(losses)

        acc = valid(epoch, net, criterion, optimizer, valid_loader)
        acc = acc.item()
        accs.append(acc)
        print(f'acc{epoch}:{acc}, train_loss:{losses}')

        log = {
            'train_losses': loss_list,
            "accs": accs,
            'epochs': list(range(epoch+1)),
        }
        with open(pathLog/f'{args.optim}_{args.lr}_{args.seed}_{args.dataset}.json', 'w') as fp:
            json.dump(log, fp)
    print("total time: ", time.time()-start)

def train(epoch, net, criterion, optimizer, train_loader):
    net.train()

    train_loss = 0
    # correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
       
       inputs = inputs.to('cuda')
       targets = targets.to('cuda')
       outputs = net(inputs)
       loss = criterion(outputs, targets)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       train_loss += loss
       total += targets.size(0)

    return train_loss/total

def valid(epoch, net, criterion, optimizer, valid_loader):
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    net.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):

            inputs = inputs.to('cuda')
            targets = targets.to('cuda')
            outputs = net(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            # top5.update(acc5[0], inputs.size(0))
            # _, predicted = outputs.max(1)
            #    print(f'batchIDX:{batch_idx}, loss: {loss}')

            #  optStep = optimizer.state_dict()['state'][0]['step']
            #  cumm = optimizer.state_dict()['state'][0]['cumm']
            #  gai = optimizer.state_dict()['state'][0]['gai']

            #  print(f"From Rank: {train_rank}:")
            #  print(f'step:batchIDX:{optStep}:{batch_idx+1}, weight: {net.module.fc.weight.item()}\ngrad: {net.module.fc.weight.grad.item()}, cumm:{cumm}\ngai: {gai}')
            # total += targets.size(0)
            #    if batch_idx==0 and epoch==0:
            #        print(f'batch size: {targets.shape[0]}')
            # correct += predicted.eq(targets).sum().item()
        return top1.avg

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