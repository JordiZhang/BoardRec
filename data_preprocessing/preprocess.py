import pandas as pd

data = pd.read_csv('../data_collection/bgg_collections.csv')

n_users_bef = data['username'].nunique()
n_games_bef = data['name'].nunique()

# filter out collectors and retailers
user_collection_size = data.groupby('username').size().sort_values(ascending=False)
data = data[data['username'].isin((user_collection_size < 500).index)].copy()

while True:
    # filter out small games, 5-core filter
    game_player_count = data.groupby('name').size().sort_values(ascending=False)
    mask1 = game_player_count >= 5
    try:
        mask1.value_counts()[False]
    except KeyError:
        break
    game_player_count = game_player_count[mask1]
    data = data[data['name'].isin(game_player_count.index)].copy()

    # filter out users new users, 5-core filter
    user_collection_size = data.groupby('username').size().sort_values(ascending=False)
    mask2 = user_collection_size >= 5
    try:
        mask2.value_counts()[False]
    except KeyError:
        break
    user_collection_size = user_collection_size[mask2]
    data = data[data['username'].isin(user_collection_size.index)].copy()


n_users_aft = data['username'].nunique()
n_games_aft = data['name'].nunique()

print(f'Before Filtering: {n_users_bef} Users, {n_games_bef} Games')
print(f'After Filtering: {n_users_aft} Users, {n_games_aft} Games')

data.to_csv('5-core_filtered_bgg.csv', index=False)