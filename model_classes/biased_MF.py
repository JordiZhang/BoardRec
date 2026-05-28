import torch
from torch import nn

class BiasedMF(nn.Module):
    def __init__(self, n_users, n_games, latent_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.game_embedding = nn.Embedding(n_games, latent_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.game_bias = nn.Embedding(n_games, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, users, games):
        pred = (self.user_embedding(users) * self.game_embedding(games)).sum(dim=1)
        pred += self.user_bias(users).squeeze() + self.game_bias(games).squeeze() + self.global_bias
        return pred