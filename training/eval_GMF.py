from model_classes.GMF import GMF
from utils.datasets import BGGDatasetTriplets, BGGDatasetTripletsEval
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

neg_samples = 999

with open('../utils/user_items.pkl', 'rb') as f:
    user_items = pickle.load(f)

train = BGGDatasetTriplets(pd.read_csv('../data_preprocessing/train.csv'), user_items)
train_loader = DataLoader(train, batch_size=4096, shuffle=True)
print('Train Data Loaded')

test = BGGDatasetTripletsEval(pd.read_csv('../data_preprocessing/test.csv'), train.n_games,
                              neg_samples, user_items)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
print('Test Data Loaded')

model = GMF(train.n_users, train.n_games, 10).to(device)
print(model)

checkpoint = torch.load('../trained_models/GMF_best_hit10.pth')
model.load_state_dict(checkpoint)

model.eval()

hit10 = []
auc_track = []
with torch.no_grad():
    for users, items in test_loader:
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