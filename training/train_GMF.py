import numpy as np
from utils.datasets import BGGDatasetTriplets, BGGDatasetEval
from model_classes.GMF import GMF
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import pickle


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

neg_samples = 499

with open('../utils/user_item.pkl', 'rb') as f:
    user_items = pickle.load(f)

train = BGGDatasetTriplets('../data_preprocessing/train.csv', user_items)
train_loader = DataLoader(train, batch_size=4096, shuffle=True)
test = BGGDatasetEval('../data_preprocessing/test.csv', train.n_games,
                      neg_samples, user_items)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
valid = BGGDatasetEval('../data_preprocessing/validation.csv', train.n_games,
                       neg_samples, user_items)
valid_loader = DataLoader(valid, batch_size=1, shuffle=False)
print('Data loaded')

model = GMF(train.n_users, train.n_games, 10).to(device)
print(model)

# BPR loss
criterion = nn.LogSigmoid()

optimizer = torch.optim.Adam(model.parameters())

n_epochs = 10

best_hit10 = [0]
best_auc = [0]
for epoch in range(n_epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
    # training
    for users, pos, neg in loop:
        users = users.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        pred_pos = model(users, pos)
        pred_neg = model(users, neg)

        pred_tot = pred_pos - pred_neg

        loss = -criterion(pred_tot).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # validation
    model.eval()
    hit10 = []
    auc_track = []
    with torch.no_grad():
        for users, items in valid_loader:
            users = users.squeeze().to(device)
            games = items.squeeze().to(device)

            predictions = model(users, games)
            rank = predictions.argsort(descending=True)
            if 0 in rank[0:10]:
                hit10.append(1)
            else:
                hit10.append(0)

            idx = (rank == 0).nonzero(as_tuple=True)[0].item()
            auc = 1 - idx/neg_samples
            auc_track.append(auc)

        hit10 = np.mean(hit10)
        auc_track = np.mean(auc_track)

        print(f'Hit@10: {hit10}, AUC: {auc_track}')

        if best_hit10[-1] < hit10:
            best_hit10.append(hit10)
            torch.save(model.state_dict(), '../trained_models/GMF_best_hit10.pth')

        if best_auc[-1] < auc_track:
            best_auc.append(auc_track)
            torch.save(model.state_dict(), '../trained_models/GMF_best_auc.pth')





