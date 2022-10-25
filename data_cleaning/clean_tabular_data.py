import pandas as pd

df = pd.read_csv("/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/Products.csv", lineterminator="\n")

# Convert str type prices to float64
df["price"] = df["price"].str.strip('£')
df["price"] = df["price"].str.replace(',', '')
df["price"] = df["price"].astype('float64')

# Write to csv file
df.to_csv('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/cleaned_products.csv')