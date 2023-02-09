import pandas as pd
import numpy as np
import os

def load_dataset(file_name):
    return pd.read_csv(file_name)

def remove_missing_values(dataset):
    df=dataset.copy()
    return df.dropna()

def remove_not_years(dataset):
    df = dataset.copy()
    df.drop(df[df['Year'] <1900].index, inplace = True)
    return df
def remove_any_wine_not_whithorred(dataset,col):
    df=dataset.copy()
    cln =df[(df[col] =='Red') | (df[col] =='White')]
    return cln

def clean_alcohol(dataset):
    df = dataset.copy()
    df.drop(df[df['Alcohol'] >25].index, inplace=True)
    return df

file_name = 'C:\develop\DAproject/WineQuality.csv'
raw_dataset = load_dataset(file_name)
cln_dataset = remove_missing_values(raw_dataset)
cln_dataset=remove_not_years(cln_dataset)
cln_dataset=remove_any_wine_not_whithorred(cln_dataset,'Category')
cln_dataset=clean_alcohol(cln_dataset)
replace_map={'White':2,'Red':1}
cln_dataset.replace(replace_map,inplace=True)
cln_dataset['Score']=cln_dataset['Score'].str.findall('\d+').str[0].astype('float')
cln_dataset=cln_dataset.drop(columns=["Unnamed: 0","Bottle"], axis=1)
cln_dataset['Price']=cln_dataset['Price'].str.findall('\d+').str[0].astype('Int64')
cln_dataset.to_csv('C:\develop\DAproject/CleanWineQuality.csv')



