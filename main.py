import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from models import create_model
from dataset import SleepApneaDataModule, EmbeddedDataModule
from utils import open_config_file, train_model, evaluate_model, test_model, epoch_time, plot_curves

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config/default.json", metavar='N', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

print('------------ Options -------------')
for k, v in vars(params).items():
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

if params.discrete_transform:
    data_module = EmbeddedDataModule(params)
else:
    data_module = SleepApneaDataModule(params)
data_module.setup()
train_dataset, val_dataset, test_dataset = data_module.train_dataset, data_module.val_dataset, data_module.test_dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)


### TRAINING

model = create_model(params)
optimizer = optim.Adam(model.parameters(), lr=params.lr)
if params.lr_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step, gamma=0.1)
model = model.cuda()

if params.loss_weighting:
    criterion = nn.BCELoss(reduction='none')
else:
    criterion = nn.BCELoss()
criterion = criterion.cuda()

best_valid_loss = float('inf')
best_valid_acc = 0.0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_accuracies_pp, val_accuracies_pp = [], []

threshold = params.threshold

for epoch in range(params.nepochs):

    start_time = time.time()
    if params.post_process:
        train_loss, train_acc, train_acc_pp = train_model(epoch+1, model, train_loader, optimizer, criterion, params, threshold)
        train_accuracies_pp.append(train_acc_pp)
    else:
        train_loss, train_acc = train_model(epoch+1, model, train_loader, optimizer, criterion, params, threshold)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    if epoch % params.eval_every == 0:
        if params.post_process:
            valid_loss, valid_acc, valid_acc_pp = evaluate_model(epoch+1, model, val_loader, criterion, params, threshold)
            val_accuracies_pp.append(valid_acc_pp)
        else:
            valid_loss, valid_acc = evaluate_model(epoch+1, model, val_loader, criterion, params, threshold)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)

        if params.tracking == 'val_loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_model.pt')

        elif params.tracking == 'val_acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), 'best_model.pt')

    if params.lr_scheduler:
        scheduler.step()

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'End of epoch {epoch+1} / {params.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s')
    # print(f'Train loss: {np.round(train_loss,6)} \t Train acc: {np.round(train_acc,4)}')
    # print(f'Val loss: {np.round(valid_loss,6)} \t Val acc: {np.round(valid_acc,4)}\n')
    print(f'Train loss: {np.round(train_loss,6)} \t Train acc: {np.round(train_acc,4)} \t Train acc pp: {np.round(train_acc_pp,4)}')
    print(f'Val loss: {np.round(valid_loss,6)} \t Val acc: {np.round(valid_acc,4)} \t Val acc pp: {np.round(valid_acc_pp,4)}\n')

bvl = np.round(best_valid_loss,6)
bvl_acc = np.round(val_accuracies[val_losses.index(best_valid_loss)],4)
if params.post_process:
    bvl_acc_pp = np.round(val_accuracies_pp[val_losses.index(best_valid_loss)],4)
bacc = np.round(np.max(val_accuracies),4)
print(f'End of training: best val loss = {bvl} | associated val_acc = {bvl_acc}, val_acc_pp = {bvl_acc_pp} | best val acc = {bacc}\n')

plot_curves(train_losses, train_accuracies, val_losses, val_accuracies, params)
if params.post_process:
    plot_curves(train_losses, train_accuracies_pp, val_losses, val_accuracies_pp, params)

### TESTING

params = open_config_file(args.config)
print('Beginning testing...')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
# load best weights from training (based on params.tracking value)
best_model = create_model(params)
best_model.load_state_dict(torch.load('best_model.pt'))
best_model = best_model.cuda()
best_model.eval()

test_predictions_df = test_model(best_model, test_loader, params, threshold=threshold)
test_predictions_df.to_csv(f'test_predictions.csv', index=False)
print('done')
