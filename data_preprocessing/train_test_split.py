import pandas as pd
import pickle

data = pd.read_csv('5-core_filtered_bgg.csv')[['username', 'name']]

user2idx = {user: idx for idx, user in enumerate(data['username'].unique())}
game2idx = {game: idx for idx, game in enumerate(data['name'].unique())}

# index mappings for users and games
with open('../utils/user2idx.pkl', 'wb') as f:
    pickle.dump(user2idx, f)

with open('../utils/game2idx.pkl', 'wb') as f:
    pickle.dump(game2idx, f)

data['username'] = data['username'].map(user2idx)
data['name'] = data['name'].map(game2idx)

# user collections, i.e. which games does each user own
user_items = data.groupby('username')['name'].apply(set).to_dict()
print(user_items)
with open('../utils/user_items.pkl', 'wb') as f:
    pickle.dump(user_items, f)

# train, test, validation
test = data.groupby('username').sample(n=1, random_state=48)
train = data.drop(test.index)

validation = train.groupby('username').sample(n=1, random_state=84)
train = train.drop(validation.index)

# save
test.to_csv('test.csv', index=False)
train.to_csv('train.csv', index=False)
validation.to_csv('validation.csv', index=False)