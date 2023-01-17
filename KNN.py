import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer


def split_to_train_and_test(dataset, test_ratio, rand_state):
    X = dataset.drop(columns="Score", axis=1)
    y = dataset["Score"]
    X_train, X_test, y_train, y_test = train_test_split(X.iloc[1:], y.iloc[1:], test_size=test_ratio, random_state=rand_state)
    return X_train, X_test, y_train, y_test

def calc_evaluation_val(eval_metric, y_test, y_predicted):
    if (eval_metric == 'accuracy'):
        return metrics.accuracy_score(y_test, y_predicted)

    if (eval_metric == 'precision'):
        return metrics.precision_score(y_test, y_predicted)

    if (eval_metric == 'f1'):
        return metrics.f1_score(y_test, y_predicted)

    if (eval_metric == 'recall'):
        return metrics.recall_score(y_test, y_predicted)

    if (eval_metric == 'confusion_matrix'):
        return metrics.confusion_matrix(y_test, y_predicted)

def find_best_k_for_KNN(X_train, y_train):
    n={'n_neighbors':[2]}
    clf=GridSearchCV(KNeighborsClassifier(),n,scoring=make_scorer(metrics.f1_score,greater_is_better=True))
    clf.fit(X_train, y_train)
    best_K=clf.best_params_['n_neighbors']
    return best_K


data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")

df=data.copy()
X_train, X_test, y_train, y_test= split_to_train_and_test(df, 0.2, 42)

k=find_best_k_for_KNN(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = calc_evaluation_val('accuracy',y_test, y_pred)
print("Accuracy:", accuracy)


#Once you have a well-performing model, you can use it to classify new wine samples based on their attributes.
#for example:
#new_wine = [["Château Cos d'Estournel 2019 G d'Estournel (Médoc)",39,13.5,750,1,"Médoc, Bordeaux, France","Bordeaux-style Red Blend","Château Cos d'Estournel"]]
#prediction = knn.predict(new_wine)
#print(prediction)
