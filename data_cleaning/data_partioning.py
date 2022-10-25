from itertools import count
import shutil
import os
import pandas as pd

df = pd.read_csv('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/indexed_data.csv', lineterminator="\n")

classification_indices = {"Home & Garden": 0, 
"Baby & Kids Stuff": 1, 
"DIY Tools & Materials": 2, 
"Music, Films, Books & Games": 3, 
"Clothes, Footwear & Accessories": 4, 
"Other Goods": 5, 
"Sports, Leisure & Travel": 6, 
"Health & Beauty": 7,
"Appliances": 8,
"Computers & Software": 9,
"Office Furniture & Equipment": 10,
"Video Games & Consoles": 11,
"Phones, Mobile Phones & Telecoms": 12}

category_names = ['baby_kids', 'diy_garden', 'music_items', 'clothes_accessories', 'other', 'sports_travel', 'health_beauty', 'appliances', 'computers_software', 'office', 'games_consoles', 'phones_telecoms']
    
home_garden = df[df['category_index'] == 0].id
baby_kids = df[df['category_index'] == 1].id
diy_garden = df[df['category_index'] == 2].id
music_items = df[df['category_index'] == 3].id
clothes_accessories = df[df['category_index'] == 4].id
other = df[df['category_index'] == 5].id
sports_travel = df[df['category_index'] == 6].id
health_beauty = df[df['category_index'] == 7].id
appliances = df[df['category_index'] == 8].id
computers_software = df[df['category_index'] == 9].id
office = df[df['category_index'] == 10].id
games_consoles = df[df['category_index'] == 11].id
phones_telecoms = df[df['category_index'] == 12].id