import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from KNN import replace_score

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")
df=data.copy()
df=replace_score(df)
#data['Price']=data['Price'].astype(str)[0].astype('float')
#['Score']=data['Score'].str.findall('\d+').str[0].astype('float')
#replace_map={'White':2,'Red':1}
#data.replace(replace_map, inplace=True)

# columns = ['From','Variety','Winery']
# le = preprocessing.LabelEncoder()
# for col in columns:
#     data[col] = le.fit_transform(data[col])
#
# X = data.drop(columns=["Unnamed: 0","Name","Score"], axis=1)
# y = data['Score']

columns = ['From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["Unnamed: 0","Name","Score"], axis=1)
y = df["Score"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
decisionTree = RandomForestClassifier()
#decisionTree = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10)

# Train the classifier on the training set
decisionTree.fit(X_train, y_train)

# Use the model to make predictions on the test set
predicted_quality = decisionTree.predict(X_test)

# Print the predicted quality
print("Predicted quality: ", predicted_quality)

accuracy = accuracy_score(y_test, predicted_quality)
print("Accuracy: ", accuracy)