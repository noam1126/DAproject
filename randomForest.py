import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from KNN import replace_score

#The code performs the following steps:
# 1.	Loads the wine quality dataset from the CSV file into the pandas dataframe. Later, replaces the "Score" column in the dataframe with binary values (0 or 1) based on whether the score is less than or equal to 95.
# 2.	Encodes the categorical columns "From", "Variety", and "Winery" to numbers so we can use the data properly.
# 3.	Trains the classifier on the training dataset.
# 4.	Uses the trained classifier to make predictions on the test dataset.
# 5.	Evaluates the accuracy of the predictions using accuracy_score method.
# 6.	Finally, the code outputs the predicted wine quality and the accuracy of the predictions.
# In addition, the accuracy here was very high too.

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")
df=data.copy()
df=replace_score(df)

columns = ['From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["Unnamed: 0","Name","Score"], axis=1)
y = df["Score"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
randomForest = RandomForestClassifier()

# Train the classifier on the training set
randomForest.fit(X_train, y_train)

# Use the model to make predictions on the test set
predicted_quality = randomForest.predict(X_test)

# Print the predicted quality
print("Predicted quality: ", predicted_quality)

accuracy = accuracy_score(y_test, predicted_quality)
print("Accuracy: ", accuracy)