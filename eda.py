import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

#%matplotlib inline

#file_name='C:\develop\DAproject/CleanWineQuality.csv'
#wine=load_csv(file_name)
#wine.to_csv('C:\develop\DAproject/EdaWineQuality.csv')
def load_csv(file_name):
    return pd.read_csv(file_name,index_col=0,header=0,sep=',')


file_name='C:\develop\DAproject/EdaWineQuality.csv'
wine=load_csv(file_name)
wine.to_csv('C:\develop\DAproject/EdaWineQuality.csv')

wine['Price']=wine['Price'].str.findall('\d+').str[0].astype('float')
wine['Score']=wine['Score'].str.findall('\d+').str[0].astype('float')

#change score to category:
bins = [80,85,90,95,100]
labels = [8,8.5,9,9.5]
wine['Score_binned'] = pd.cut(wine['Score'], bins=bins, labels=labels)
wine.Score_binned.describe()

#change category to 1=red, 2=white:
replace_map={'White':2,'Red':1}
wine.replace(replace_map, inplace=True)
wine.head()

columns = ['From','Variety','Winery']
le = preprocessing.LabelEncoder()
for col in columns:
    wine[col] = le.fit_transform(wine[col])
wine = wine.drop(columns=["Unnamed: 0"], axis=1)

#גרף עמודות של היחס בין כמות היינות לדירוגים:
sns.countplot(x='Score_binned',data=wine)
plt.title('amount of wines according to the score:')
#גרף עמודות של היחס בין כמות היינות לדירוגים-תמצות:
#sns.countplot(x='Score',data=wine)
#plt.title('amount of wines according to the score binned:')

#גרף בערך של היחס בין המחיר לדירוג
sns.lmplot(x='Score',y='Price',data=wine)
plt.title('the connection between the price to the score:')

#גרף בערך של היחס בין האחוז אלכוהול למחיר
#sns.lmplot(x='Alcohol',y='Price',data=wine)
#plt.title('the connection between the price to the AVB:')

#create heatmap:
plt.figure(figsize=(10,7))
sns.heatmap(wine.corr(),color="k",annot=True)

#sns.swarmplot(x="Score_binned",y="Alcohol",data=wine)
sns.stripplot(data=wine,x="Score_binned",y='Alcohol')

#wine.groupby('Category')['Score'].mean().plot.line()
#plt.ylabel('score')

plt.show()

