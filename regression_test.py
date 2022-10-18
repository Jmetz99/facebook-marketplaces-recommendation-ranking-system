import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools

df = pd.read_csv('cleaned_products.csv', lineterminator="\n")

prices = df['price']

df = df.drop('price', axis=1)
df = df.drop('Unnamed: 0.1', axis=1)
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('id', axis=1)
df = df.drop('category', axis=1)

# Split sample features and prices into training and test
training_features = df.sample(frac=0.7, random_state=0)
test_data = df.drop(training_features.index)

training_outputs = prices.sample(frac=0.7, random_state=0)
test_outputs = prices.drop(training_outputs.index)

product_names = training_features['product_name'].tolist()
product_description = training_features['product_description'].tolist()
product_location = training_features['location'].tolist()

vectorizer = TfidfVectorizer()

names_response = vectorizer.fit_transform(product_names)

print(names_response)


# description_response = vectorizer.fit_transform(product_description)
# for col in description_response.nonzero()[1]:
#     print (feature_names[col], ' - ', description_response[0, col])

# location_response = vectorizer.fit_transform(product_location)
# for col in location_response.nonzero()[1]:
#     print (feature_names[col], ' - ', location_response[0, col])


class LinearRegression:
    def __init__(self, n_features: int): # initalise parameters
        np.random.seed(10)
        self.W = np.random.randn(n_features, 1) ## randomly initialise weight
        self.b = np.random.randn(1) ## randomly initialise bias
        
    def __call__(self, X):
        ypred = np.dot(X, self.W) + self.b
        return ypred # return prediction
    
    def update_params(self, W, b):
        self.W = W 