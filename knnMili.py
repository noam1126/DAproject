import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")

data['Price']=data['Price'].str.findall('\d+').str[0].astype('float')
data['Score']=data['Score'].str.findall('\d+').str[0].astype('float')
replace_map={'White':2,'Red':1}
data.replace(replace_map, inplace=True)

columns = ['Name','From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    data[col] = le.fit_transform(data[col])

data.describe()

X = data.drop(columns='Score', axis=1)
y = data['Score']

#wanted_columns = ['Year', 'Score', 'Category','Alcohol','Price']
# Select only the wanted columns
#X = data[wanted_columns]
#y = data['Score']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Train the classifier on the training set
clf.fit(X_train, y_train)

#Create a feature vector for the wine you want to predict the score for
#wine_features = []

random_features = [random.uniform(0, 1) for _ in range(X.shape[1])]

predicted_quality = clf.predict(np.array(random_features).reshape(1,-1))

# Convert the feature vector to a numpy array
#wine_features = np.array(wine_features)

# Use the model to make a prediction
#predicted_quality = clf.predict(wine_features.reshape(1,-1))

# Print the predicted quality
print("Predicted quality: ", predicted_quality)

accuracy = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy: ", accuracy)
