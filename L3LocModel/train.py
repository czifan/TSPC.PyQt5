from model import *
from dataset import *
from utils import *
import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import os 
import matplotlib.pyplot as plt 
from prettytable import PrettyTable
import shutil 
from copy import deepcopy
import thop

def train(args, epoch, model, dl, criterion, optimizer, printer):
    model.train()
    records = {"Loss": []}
    y_true, y_pred = [], []
    optimizer.zero_grad()
    for idx, data in enumerate(dl):
        X = data["X"].float().to(args.device)
        y = data["y"].float().to(args.device)

#         if epoch == 1 and idx == 0:
#             flops, params = thop.profile(model, inputs=(X,))
#             printer(f"Flops={flops/1e9:.2f} G\tParams={params/1e6:.2f} M")

        #p = model(X)
        p = model(X, [len(y[i]) for i in range(y.shape[0])])
        loss = criterion(p, y) / args.batch_size
        #print(loss.item())
        records["Loss"].append(loss.item() * args.batch_size)
        loss.backward()
        
        if ((idx + 1)%args.batch_size) == 0 or (idx == len(dl)-1):
            optimizer.step()
            optimizer.zero_grad()
        
        p = torch.softmax(p, dim=1)
        y_true.append(torch.argmax(y, dim=1).detach().cpu().numpy())
        y_pred.append(torch.argmax(p, dim=1).detach().cpu().numpy())
        
    y_pred = np.stack(y_pred, axis=0)
    y_true = np.stack(y_true, axis=0)
    records["Acc@1"] = topK_accuracy(y_pred, y_true, K=1)
    records["Acc@3"] = topK_accuracy(y_pred, y_true, K=3)
    records["Acc@5"] = topK_accuracy(y_pred, y_true, K=5)
    context = f"[TRAIN]\tEpoch: {epoch}\t"
    for key, value in records.items():
        context += f"{key}: {np.mean(value):.4f}\t"
    printer(context)
    return records

def evaluate(args, epoch, model, dl, criterion, printer, split):
    model.eval()
    records = {"Loss": []}
    y_true, y_pred = [], []
    for idx, data in enumerate(dl):
        with torch.no_grad():
            X = data["X"].float().to(args.device)
            y = data["y"].float().to(args.device)
            
            #p = model(X)
            p = model(X, [len(y[i]) for i in range(y.shape[0])])
            loss = criterion(p, y)

            p = torch.softmax(p, dim=1)
            y_true.append(torch.argmax(y, dim=1).detach().cpu().numpy())
            y_pred.append(torch.argmax(p, dim=1).detach().cpu().numpy())
            records["Loss"].append(loss.item())
        
    y_pred = np.stack(y_pred, axis=0)
    y_true = np.stack(y_true, axis=0)
    records["Acc@1"] = topK_accuracy(y_pred, y_true, K=1)
    records["Acc@3"] = topK_accuracy(y_pred, y_true, K=3)
    records["Acc@5"] = topK_accuracy(y_pred, y_true, K=5)
    context = f"[{split}]\tEpoch: {epoch}\t"
    for key, value in records.items():
        context += f"{key}: {np.mean(value):.4f}\t"
    printer(context)
    return records

def main(args, printer):
    model = eval(args.model)(args)
    if os.path.isfile(args.resume):
        model_state_dict = model.state_dict()
        suc = 0
        checkpoint = torch.load(args.resume)
        for key, value in checkpoint.items():
            new_key = key
            if new_key in model_state_dict and model_state_dict[new_key].shape == value.shape:
                model_state_dict[new_key] = value 
                suc += 1
        model.load_state_dict(model_state_dict)
        printer(f'Loaded weight from {args.resume}: {suc}/{len(model_state_dict)}')
    model = model.to(args.device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                        T_0=50, T_mult=1, 
                                                                        eta_min=args.lr/100.0, 
                                                                        last_epoch=-1, verbose=True)
    criterion = GaussianLoss().to(args.device)
    #criterion = CrossEntropyLoss().to(args.device)

    train_dl = DataLoader(eval(args.dataset)(args.data_dir, "train", args.train_split_file, printer,
                                       window_level=args.window_level, window_width=args.window_width,
                                       target_size=args.target_size, sigma=args.sigma),
                         batch_size=1, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_dl = DataLoader(eval(args.dataset)(args.data_dir, "valid", args.valid_split_file, printer,
                                           window_level=args.window_level, window_width=args.window_width,
                                           target_size=args.target_size, sigma=args.sigma),
                             batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_dl = DataLoader(eval(args.dataset)(args.data_dir, "test", args.test_split_file, printer,
                                           window_level=args.window_level, window_width=args.window_width,
                                           target_size=args.target_size, sigma=args.sigma),
                             batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    save_valid_records = None
    save_test_records = None
    for epoch in range(1, args.epochs+1):
        train(args, epoch, model, train_dl, criterion, optimizer, printer)
        valid_records = evaluate(args, epoch, model, valid_dl, criterion, printer, "VALID")
        test_records = evaluate(args, epoch, model, test_dl, criterion, printer, "TEST")
                                
        if save_valid_records is None or save_valid_records[args.metric] < valid_records[args.metric]:
            save_valid_records = valid_records
            save_test_records = test_records
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'last.pth'))
        printer('')
        
        lr_scheduler.step()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='%s_%s_%s_%s_%s')
    parser.add_argument('--model', type=str, default='L3LocModel')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--N_neighbor', type=int, default=2)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--window_level', type=int, default=250)
    parser.add_argument('--window_width', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='L3LocDataset')
    
    parser.add_argument('--data_dir', type=str, default='Data/L3LocData')
    parser.add_argument('--train_split_file', type=str, default='Data/train_L3.txt')
    parser.add_argument('--valid_split_file', type=str, default='Data/valid_L3.txt')
    parser.add_argument('--test_split_file', type=str, default='Data/test_L3.txt')
    parser.add_argument('--target_size', type=int, default=96)
    
    parser.add_argument('--metric', type=str, default='Acc@3')
    parser.add_argument('--init_seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='Results')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = get_args()
    args.exp = args.exp % (args.model, args.backbone, args.pretrained, args.N_neighbor, args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.exp)
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = build_logging(os.path.join(args.output_dir, 'log.log'))
    printer = logger.info
    print_args(args, printer)
    setup_seed(args.init_seed)
    main(args, printer)