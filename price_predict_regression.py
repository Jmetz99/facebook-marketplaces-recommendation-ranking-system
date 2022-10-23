import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from matplotlib import pyplot as plt
from scipy import sparse

df = pd.read_csv('cleaned_products.csv', lineterminator="\n")

prices = df['price']

# Drop unused features
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

# Isolate features
product_names = training_features['product_name']
product_description = training_features['product_description']
product_location = training_features['location']

product_test = training_features

# Get TF-IDF scores
vectorizer = TfidfVectorizer()
description_response = vectorizer.fit_transform(product_description).toarray()
names_response = vectorizer.fit_transform(product_names).toarray()
location_response = vectorizer.fit_transform(product_location).toarray()

responses_concat = np.concatenate((names_response, description_response, location_response), axis=1)
sparse_responses = sparse.csr_matrix(responses_concat)

# Regress and predict 
model = linear_model.LinearRegression()
model.fit(sparse_responses, training_outputs)
y_pred = model.predict(sparse_responses)

# Evaluate MSE of linear model(=8.3131305359371)
print(metrics.mean_squared_error(training_outputs, y_pred))