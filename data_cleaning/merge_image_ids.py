import pandas as pd
from PIL import Image
from numpy import asarray
import os

cleaned_products_df = pd.read_csv('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/cleaned_products.csv', lineterminator="\n")
cleaned_products_df = cleaned_products_df.drop('Unnamed: 0.1', axis=1)
cleaned_products_df = cleaned_products_df.drop('Unnamed: 0', axis=1)
cleaned_products_df = cleaned_products_df.rename({'id': 'product_id'}, axis=1)

df = pd.read_csv('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/Images.csv')

merged_df = pd.merge(cleaned_products_df, df, on='product_id')
merged_df = merged_df.drop('Unnamed: 0', axis=1)

merged_df.to_csv('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/full_df.csv')
