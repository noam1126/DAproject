import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from KNN import replace_score
from sklearn import tree

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")
df=data.copy()
# data['Price']=data['Price'].str.findall('\d+').str[0].astype('float')
# data['Score']=data['Score'].str.findall('\d+').str[0].astype('float')
# replace_map={'White':2,'Red':1}
# data.replace(replace_map, inplace=True)

# columns = ['From','Variety','Winery']
# le = preprocessing.LabelEncoder()
# for col in columns:
#     data[col] = le.fit_transform(data[col])

# X = data.drop(columns=["Unnamed: 0","Name","Score"], axis=1)
# y = data['Score']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#wanted_columns = ['Year', 'Score', 'Category','Alcohol','Price']
#X = data[wanted_columns]
#y = data['Score']

df=replace_score(df)
columns = ['From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["Unnamed: 0","Name","Score"], axis=1)
y = df["Score"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


#param_grid = {'max_depth': range(1, 11)}
#decisionTree = DecisionTreeClassifier()
#grid_search = GridSearchCV(decisionTree, param_grid, cv=5, scoring=make_scorer(accuracy_score))
#grid_search.fit(X_train, y_train)
#print("Best maximum depth: ", grid_search.best_params_)
#print("Best score: ", grid_search.best_score_)

#decisionTree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20)
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
