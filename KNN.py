import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer
from sklearn import preprocessing
import math

def find_best_k_for_KNN(X_train, y_train):
    k=[]
    sqr=math.sqrt(X_train.shape[0])
    for i in range(1,int(sqr),2):
        k.append(i)
    n={'n_neighbors':k}
    clf=GridSearchCV(KNeighborsClassifier(),n,scoring=make_scorer(metrics.f1_score,greater_is_better=True))
    clf.fit(X_train, y_train)
    best_k=clf.best_params_['n_neighbors']
    return best_k

# def replace_score(dataset):
#     df = dataset.copy()
#     df.loc[df.Score < 93, 'Score'] = 0
#     df.loc[df.Score >= 93, 'Score'] = 1
#     return df

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")
df=data.copy()

#df=replace_score(df)

columns = ['From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])

#sqr of X_train
X = df.drop(columns=["Name","Score"], axis=1)
y = df["Score"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# knn = KNeighborsClassifier(n_neighbors=65)
# knn.fit(X_train,y_train)
#
# y_pred = knn.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

#find the best k
k=find_best_k_for_KNN(X_train, y_train)
print(k)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

