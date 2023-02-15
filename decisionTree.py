import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from KNN import replace_score
from sklearn import tree

# At the start we use the Decision Tree method to classify wine quality based on various features. The code performs the following steps:
# 1.	Loads the wine quality dataset from the CSV file into the pandas dataframe. Later, replaces the "Score" column in the dataframe with binary values (0 or 1) based on whether the score is less than or equal to 95.
# 2.	Encodes the categorical columns "From", "Variety", and "Winery" to numbers so we can use the data properly.
# 3.	Then, train the Decision Tree Classifier model on the training data.
# 4.	Predict the quality of random features.
# 5.	In the end, calculate the accuracy of the model on the training data and testing data.
# In conclusion, the accuracy here was very high so we wanted to check more methods.

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")
df=data.copy()
df=replace_score(df)
columns = ['From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["Unnamed: 0","Name","Score"], axis=1)
y = df["Score"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

decisionTree = tree.DecisionTreeClassifier()
decisionTree=decisionTree.fit(X_train, y_train)

random_features = [random.uniform(0, 1) for _ in range(X.shape[1])]
predicted_quality = decisionTree.predict(np.array(random_features).reshape(1,-1))
print("Predicted quality: ", predicted_quality)

y_pred_train=decisionTree.predict(X_train)
print('Accuracy on training data= ', accuracy_score(y_train, y_pred_train))

y_pred=decisionTree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
