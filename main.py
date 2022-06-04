import os
import pandas as pd
import numpy as np
import argparse
import random

import torch
from torch.utils.data import DataLoader
import wandb

from data import IshiharaDataset, Food11Dataset
from model import OurModel, UNet




###
parser = argparse.ArgumentParser()
parser.add_argument("--log", default='tmp')
parser.add_argument("--seed", default=0)
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

### seed 고정
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.dataset == "ishihara":
    training_data = IshiharaDataset(train_flag=True)
    testing_data = IshiharaDataset(train_flag=False)
elif args.dataset == "food11":
    training_data = Food11Dataset(train_flag=True)
    testing_data = Food11Dataset(train_flag=False)

training_loader = DataLoader(training_data, batch_size=4, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 15

wandb.init(
    project='moonshot',
    name=f'{args.dataset}-{args.log}',
    entity='noparkee',
    )
print('Results are now reporting to WANDB (wandb.ai)')

#our_model = OurModel(device, training_data.num_class).to(device)
our_model = UNet(device, training_data.num_class).to(device)
opt = torch.optim.Adam(our_model.parameters())

our_model.train()
for e in range(EPOCH):
    epoch_loss = 0
    for i, data in enumerate(training_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        loss = our_model(x, y, i)

        opt.zero_grad()
        loss["loss"].backward()
        opt.step()

        wandb.log(loss)

        epoch_loss += loss["loss"].item()
        if i % 100 == 99 or i == 0:
            if i == 0:
                print(f'[{e + 1}, {i + 1:5d}] loss: {epoch_loss:.3f}')
            else:
                print(f'[{e + 1}, {i + 1:5d}] loss: {epoch_loss / 100:.3f}')
            epoch_loss = 0.0
    
    ### test
    ### 매 epoch 마다
    our_model.eval()
    correct, total = 0, 0
    for test_data in testing_loader:
        x, y = test_data
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            c, t = our_model.evaluate(x, y)
            correct += c
            total += t

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    if not os.path.exists(f'results/models/{args.log}'):
        os.makedirs(f'results/models/{args.log}')
    torch.save(our_model.state_dict(), f'results/models/{args.log}/{e+1}epoch.pt')
