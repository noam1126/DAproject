import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")

X = data.drop("Score", axis=1)
y = data["Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5) # specify the value of k
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#try different value of k,
#test the accuracy and choose the best one

#Once you have a well-performing model, you can use it to classify new wine samples based on their attributes.
#for example:
new_wine = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
prediction = knn.predict(new_wine)
print(prediction)
