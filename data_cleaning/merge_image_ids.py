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

file_names = os.listdir('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/resized_images') 

ids = []
arrays = []

for file in file_names:
    id = file.strip('.jpg_resized.jpg')
    ids.append(id)
    img = Image.open(f'/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/resized_images/{file}')
    image_array = asarray(img)
    arrays.append(image_array)

ids_series = pd.Series(ids, name='id')
arrays_series = pd.Series(arrays, name='image_array')

image_df = pd.concat([ids_series, arrays_series], axis=1)

full_df = pd.merge(merged_df, image_df, on='id')

full_df.to_pickle('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/full_df.pickle')