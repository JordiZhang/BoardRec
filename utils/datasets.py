import random
from torch.utils.data import Dataset
import torch
import pandas as pd


class BGGDatasetTriplets(Dataset):
    def __init__(self, path, user_items):
        collections = pd.read_csv(path)
        self.n_users = collections['username'].nunique()
        self.n_games = collections['name'].nunique()
        self.interactions = torch.tensor(collections.values, dtype=torch.long)
        self.user_items = user_items

    def __len__(self):
        return len(self.interactions)

    def sample_negative(self, user, n_games):
        while True:
            game = random.randint(0, n_games - 1)
            if game not in self.user_items[user]:
                return game

    def __getitem__(self, idx):
        user, game = self.interactions[idx]
        user = user.item()
        game = game.item()
        negative = self.sample_negative(user, self.n_games)
        return user, game, negative


class BGGDatasetEval(Dataset):
    def __init__(self, path, tot_games, neg_samples, user_items):
        collections = pd.read_csv(path)
        self.interactions = torch.tensor(collections.values, dtype=torch.long)
        self.tot_games = tot_games
        self.neg_samples = neg_samples
        self.user_items = user_items

        self.samples = []

        for user, pos_game in self.interactions:
            user = user.item()
            pos_game = pos_game.item()
            negatives = set()
            while len(negatives) < self.neg_samples:
                neg = self.sample_negative(user, self.tot_games)

                if neg in negatives:
                    continue
                negatives.add(neg)

            items = torch.tensor([pos_game] + list(negatives), dtype=torch.long)
            users = torch.full((len(items),), user, dtype=torch.long)
            self.samples.append((users, items))

    def __len__(self):
        return len(self.interactions)

    def sample_negative(self, user, n_games):
        while True:
            game = random.randint(0, n_games - 1)
            if game not in self.user_items[user]:
                return game

    def __getitem__(self, idx):
        return self.samples[idx]
