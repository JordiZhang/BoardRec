from torch import nn

class GMF(nn.Module):
    def __init__(self, n_users, n_games, emb_dim):
        super().__init__()

        self.n_users = n_users
        self.n_games = n_games
        self.emb_dim = emb_dim
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.game_embedding = nn.Embedding(n_games, emb_dim)

        self.neural_net = nn.Sequential(
            nn.Linear(int(emb_dim), int(emb_dim/2)),
            nn.ReLU(),
            nn.Linear(int(emb_dim/2), 1)

        )

    def forward(self, users, games):
        user_embedding = self.user_embedding(users)
        game_embedding = self.game_embedding(games)
        inp = user_embedding * game_embedding

        out = self.neural_net(inp)
        return out.squeeze()
