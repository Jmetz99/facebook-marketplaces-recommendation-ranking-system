import pandas as pd
 
df = pd.read_csv(f'/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/full_df.csv', lineterminator="\n")

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

def index_category(category):
    for key in classification_indices.keys():
        if key in category:
            return classification_indices[key]

df['category_index'] = df['category'].map(index_category)

df.to_csv('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/indexed_data.csv')