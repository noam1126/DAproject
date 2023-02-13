import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
<<<<<<< HEAD
=======
#%matplotlib inline
>>>>>>> origin/develop

# Heatmap- the colors represent the connection between two features. The purpose of the map is when it closer to one the connection is stronger.
# Countplot- show that most of the wines get high score, so the average is 85-95.

def load_csv(file_name):
    return pd.read_csv(file_name)

file_name='C:\develop\DAproject/CleanWineQuality.csv'
wine=load_csv(file_name)
wine.to_csv('C:\develop\DAproject/EdaWineQuality.csv')
wine=load_csv('C:\develop\DAproject/EdaWineQuality.csv')

#change score to category:
bins = [80,85,90,95,100]
labels = [8,8.5,9,9.5]
wine['Score_binned'] = pd.cut(wine['Score'], bins=bins, labels=labels)
wine.Score_binned.describe()

wine.head()

columns = ['From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    wine[col] = le.fit_transform(wine[col])
<<<<<<< HEAD
wine=wine.drop(columns=["Unnamed: 0",'Unnamed: 0.1'], axis=1)
=======
wine=wine.drop(columns=["Unnamed: 0"], axis=1)
>>>>>>> origin/develop

sns.countplot(x='Score_binned',data=wine)
plt.title('amount of wines according to the score:')

sns.lmplot(x='Score',y='Price',data=wine)
plt.title('the connection between the price to the score:')

plt.figure(figsize=(10,7))
sns.heatmap(wine.corr(),color="k",annot=True)

sns.stripplot(data=wine,x="Score_binned",y='Alcohol')


plt.show()

wine.groupby('Alcohol')['Score'].mean().plot.line()
plt.ylabel('score')
plt.title('the connection between alcohol% to the score:')
plt.show()

sns.stripplot(data=wine,x="Score_binned",y='Alcohol')
plt.xlabel('score')
plt.show()

wine.groupby('Price')['Alcohol'].mean().plot.line()
plt.ylabel('Alcohol')
plt.title('the connection between the alcohol% to the price:')
plt.show()

sns.stripplot(data=wine,x="Category",y='Alcohol')
plt.xlabel('Category')
plt.title('which wine (red/white) has more alcohol% in avg?')
plt.show()

sns.histplot(data=wine['Price'])
plt.show()

wine.hist(bins=20)
plt.xlabel('alcohol')
plt.ylabel('price')

sns.pairplot(wine[['Alcohol','Score', 'Price','Year','Category']])

sns.countplot(x='Category',data=wine)
plt.title('amount of wines according to the score:')
