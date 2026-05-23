import pandas as pd

data = pd.read_csv('5-core_filtered_bgg.csv')[['username', 'name']]

test = data.groupby('username').sample(n=1, random_state=48)
train = data.drop(test.index)

validation = train.groupby('username').sample(n=1, random_state=84)
train = train.drop(validation.index)

test.to_csv('test.csv', index=False)
train.to_csv('train.csv', index=False)
validation.to_csv('validation.csv', index=False)