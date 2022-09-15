import argparse
import json
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.datasets import ImageFolder

from CONSTANTS import LABEL_MAP
from dataset import InputData
from metrics import AverageMeter, accuracy
from transform import build_transform
from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, required=False, help='input resume model path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epoch', type=int, default=50, help='training epoch')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--eval', type=bool, default=False, help='eval mode')
parser.add_argument('--data-path', type=str, default='Animals Dataset', help='dataset path')
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--output-path', type=str, default='ans')
parser.add_argument('--pretrained', type=bool, default=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

transform_labeled =  build_transform(True)
transform_val = build_transform(False)

data_path = args.data_path 
batch_size = args.batch_size
num_workers = args.workers 

train_path = (os.path.join(data_path, 'train'))
test_path = (os.path.join(data_path, 'test'))

ori_dataset = ImageFolder(
    train_path,
    transform_labeled,
    )
test_dataset = InputData(
    test_path,
    transform=transform_val)

n_valid = int(len(ori_dataset) * 0.1)
n_train = len(ori_dataset) - n_valid
valid_dataset, train_dataset = random_split(ori_dataset, 
                                        [n_valid, n_train],
                                        generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    num_workers=num_workers)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=num_workers)

if not args.eval:
    w = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if args.pretrained else None

    learning_rate = args.lr #@param
    epochs = args.epoch #@param
    loss_fn = nn.CrossEntropyLoss()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    cur_epoch = 0

    if args.resume is None:
        if args.pretrained:
            model = models.convnext_base(weights=w)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = models.convnext_base(weights=w).classifier
        else:
            model = models.convnext_base()
        model.classifier[2] = nn.Linear(1024, 22, True)
        model.to(device=device)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, threshold=1e-6)
        warmup_scheduler = GradualWarmupScheduler(optimizer, 
                                                multiplier=10, 
                                                total_epoch=10, 
                                                after_scheduler=scheduler)
        
    else:
        model = models.convnext_base()
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = models.convnext_base(weights=w).classifier
        model.classifier[2] = nn.Linear(1024, 22, True)
        print(model)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["net"])
        model.to(device=device)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr = learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer"])
        cur_epoch = checkpoint["epoch"] + 1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, threshold=1e-6)
        scheduler.load_state_dict(checkpoint["sche"])
        warmup_scheduler = GradualWarmupScheduler(optimizer, 
                                                multiplier=1, 
                                                total_epoch=10, 
                                                after_scheduler=scheduler)
        warmup_scheduler.load_state_dict(checkpoint["warmup"])

    start = time.time()
    best = 0.0
    print(f"training: batch num {len(train_loader)}")
    for i in range(cur_epoch, epochs):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')        
        t_losses = AverageMeter('Loss', ':.4e')
        t_top1 = AverageMeter('Acc@1', ':6.2f')
        t_top5 = AverageMeter('Acc@5', ':6.2f')

        for batch, (X, y) in enumerate(train_loader):
            data_time.update(time.time() - start)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).to(device)

            optimizer.zero_grad() 
            loss.backward() # 这一步即反向传播
            optimizer.step()

            acc1, acc5 = accuracy(pred, y, topk=(1, 5))
            t_losses.update(loss.item(), X.size(0))
            t_top1.update(acc1[0], X.size(0))
            t_top5.update(acc5[0], X.size(0))
        
        for batch, (X, y) in enumerate(valid_loader):
            data_time.update(time.time() - start)
            X = X.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y).to(device)

                acc1, acc5 = accuracy(pred, y, topk=(1, 5))
                losses.update(loss.item(), X.size(0))
                top1.update(acc1[0], X.size(0))
                top5.update(acc5[0], X.size(0))

        lr = optimizer.param_groups[0]['lr']
        warmup_scheduler.step(i+1, t_losses.avg) if i < 10 else None
            
        batch_time.update(time.time() - start)
        start = time.time()

        print(f"Epoch:{i + 1}: {batch_time}({batch_time.avg:.2f}), " +\
        f"{losses}({t_losses.avg:.2f}), {top1}({t_top1.avg:.2f}), {top5}({t_top5.avg:.2f}), lr: {lr}")

        if top1.avg > best and i+1 > 10:
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i,
                "sche": scheduler.state_dict(),
                "warmup": warmup_scheduler.state_dict()
            }
            torch.save(checkpoint, f"newconvnext\\{i+1}.pth")
            best = top1.avg

else:
    if args.resume is None:
        raise ValueError("You should use '--resume' to locate model path!")
    model = models.convnext_base()
    model.classifier[2] = nn.Linear(1024, 22, True)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["net"])
    model.to(device=device)
    model.eval()
    ans = []

    for X, _ in test_loader:
        X = X.to(device)
        pred = model(X)
        _, id = torch.max(pred.data, 1)
        ans.append(int(id))
    
    dicord = [str(i) for i in range(110)]
    dicord.sort()
    labels_ans = {str(dicord[i]):LABEL_MAP[l] for i, l in enumerate(ans)}

    lbs = json.dumps(labels_ans)
    with open(os.path.join(args.output_path, "convnext.json"), "w") as f:
        f.write(lbs)
