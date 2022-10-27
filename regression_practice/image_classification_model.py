import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_pickle('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/indexed_data.pickle')

X = np.array(df['image_array'])
y = np.array(df['category_index'])
 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)

log_reg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(accuracy_score(y_test, y_pred))