import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model_classes.GMF import GMF
import pickle
from utils.api_query import get_user_collection
from utils.datasets import BGGDatasetTriplets
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

with open('../utils/user_items.pkl', 'rb') as f:
    user_items = pickle.load(f)

with open('../utils/user2idx.pkl', 'rb') as f:
    user2idx = pickle.load(f)

with open('../utils/game2idx.pkl', 'rb') as f:
    game2idx = pickle.load(f)

idx2user = {idx: user for user, idx in user2idx.items()}
idx2game = {idx: game for game, idx in game2idx.items()}

n_users = len(user2idx)
n_games = len(game2idx)

# +1 to n_users for the queried user
model = GMF(n_users+1, n_games, 10).to(device)
checkpoint = torch.load('../trained_models/GMF_best_hit10.pth')

# freeze all parameters except user embedding
for name, param in model.named_parameters():
    # copy over trained model weights
    if 'user_embedding.weight' not in name:
        param = checkpoint[name]
        param.require_grad = False
    # treat user_embeddings separately
    else:
        with torch.no_grad():
            print(param)
            param[:n_users] = checkpoint[name]
            print(param)

# define new user
user = 'BoardRecTest'

if user in user2idx.keys():
    user = user2idx[user]
    games = user_items[user]
    users = [user2idx[user]]*len(games)
    collection = {'username': users, 'name': games}
    collection = pd.DataFrame(collection)

    print(collection)
    # unfinished for existing user

else:
    # get collection
    collection = get_user_collection(user)
    print(collection)

    # drop games which aren't part of training set, either too rare or new games
    mask = collection['name'].isin(game2idx.keys())
    collection = collection.drop(collection[~mask].index).reset_index(drop=True)
    print(collection)

    # change to indices
    collection['username'] = n_users
    collection['name'] = collection['name'].map(game2idx)
    print(collection)

    # add to user_items
    print(user_items[0])
    user_items[n_users] = set(collection['name'])
    print(user_items[n_users])


    triplets = BGGDatasetTriplets(collection, len(user2idx), len(game2idx), user_items)
    loader = DataLoader(triplets, batch_size=8, shuffle=True)

    # add load optimizer too
    criterion = nn.LogSigmoid()

    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    # fold-in the new user vector
    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{n_epochs}")
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
        losses.append(total_loss / len(loader))

    #plt.plot(losses)
    #plt.show()
    model.eval()

    for name, param in model.named_parameters():
        # copy over trained model weights
        if 'user_embedding.weight' in name:
            print(param)

    predictions = np.zeros(len(idx2game))
    for i, game in enumerate(idx2game.keys()):
        predictions[i] = model(torch.tensor(n_users), torch.tensor(game))

    rankings = np.argsort(-predictions)
    for i in range(100):
        print(idx2game[int(rankings[i])])