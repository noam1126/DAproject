import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
#X = data[wanted_columns]
#y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)
#wine_features = []

random_features = [random.uniform(0, 1) for _ in range(X.shape[1])]

predicted_quality = clf.predict(np.array(random_features).reshape(1,-1))
print("Predicted quality: ", predicted_quality)

accuracy = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy: ", accuracy)
