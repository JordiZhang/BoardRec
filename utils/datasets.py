import random
from torch.utils.data import Dataset
import torch
import numpy as np


class BGGDatasetTriplets(Dataset):
    def __init__(self, collections, n_users, n_games, user_items):
        self.n_users = n_users
        self.n_games = n_games
        self.interactions = torch.tensor(collections.values, dtype=torch.long)
        self.user_items = user_items
        user_items = [x for val in user_items.values() for x in list(val)]
        self.games, self.weights= np.unique(user_items, return_counts=True)
        self.weights = self.weights**0.75
        self.weights /= self.weights.sum()

    def __len__(self):
        return len(self.interactions)

    def sample_negative(self, user):
        while True:
            game = int(np.random.choice(self.games, p=self.weights))
            if game not in self.user_items[user]:
                return game

    def __getitem__(self, idx):
        user, game = self.interactions[idx]
        user = user.item()
        game = game.item()
        negative = self.sample_negative(user)
        return user, game, negative


class BGGDatasetTripletsEval(Dataset):
    def __init__(self, collections, n_games, neg_samples, user_items):
        self.interactions = torch.tensor(collections.values, dtype=torch.long)
        self.n_games = n_games
        self.neg_samples = neg_samples
        self.user_items = user_items
        user_items = [x for val in user_items.values() for x in list(val)]
        self.games, self.weights = np.unique(user_items, return_counts=True)
        self.weights = self.weights ** 0.75
        self.weights /= self.weights.sum()

        self.samples = []

        for user, pos_game in self.interactions:
            user = user.item()
            pos_game = pos_game.item()
            negatives = self.sample_negatives(user)

            items = torch.tensor([pos_game] + negatives, dtype=torch.long)
            users = torch.full((len(items),), user, dtype=torch.long)
            self.samples.append((users, items))

    def __len__(self):
        return len(self.interactions)

    def sample_negatives(self, user):
        user_games = self.user_items[user]

        negatives = set()

        while len(negatives) < self.neg_samples:

            candidates = np.random.choice(
                self.games,
                size=self.neg_samples * 2,
                p=self.weights
            )

            for game in candidates:
                if game not in user_games:
                    negatives.add(int(game))

                    if len(negatives) == self.neg_samples:
                        break

        return list(negatives)

    def __getitem__(self, idx):
        return self.samples[idx]
