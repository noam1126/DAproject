import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")

X = data.drop("Score", axis=1)
y = data["Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

k=5
knn = KNeighborsClassifier(n_neighbors=k) # specify the value of k
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#try different value of k,
#test the accuracy and choose the best one

#Once you have a well-performing model, you can use it to classify new wine samples based on their attributes.
#for example:
new_wine = [["Château Cos d'Estournel 2019 G d'Estournel (Médoc)",39,13.5,750,1,"Médoc, Bordeaux, France","Bordeaux-style Red Blend","Château Cos d'Estournel"]]
prediction = knn.predict(new_wine)
print(prediction)
