import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_pickle('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/indexed_data.pickle')

print(df.columns)

print(df['category'])
print(df['category_index'])

