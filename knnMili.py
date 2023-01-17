import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv("C:\develop\DAproject/CleanWineQuality.csv")

data['Price']=data['Price'].str.findall('\d+').str[0].astype('float')
data['Score']=data['Score'].str.findall('\d+').str[0].astype('float')
replace_map={'White':2,'Red':1}
data.replace(replace_map, inplace=True)

wanted_columns = ['Year', 'Score', 'Category','Alcohol','Price']
# Select only the wanted columns
X = data[wanted_columns]
y = data['Score']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the linear regression model
lin_reg = LinearRegression()

# Train the model on the training set
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lin_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#Evaluate the performance of the model
#from sklearn.metrics import mean_squared_error
#print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

