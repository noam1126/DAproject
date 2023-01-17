import pandas as pd
import numpy as np
import os

def load_dataset(file_name):
    return pd.read_csv(file_name)

def remove_missing_values(dataset):
    df=dataset.copy()
    return df.dropna()

def remove_duplicate_rows(dataset):
    df=dataset.copy()
    return df.drop_duplicates()

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

def replace_score(dataset):
    df = dataset.copy()
    df.loc[df.Score < 93, 'Score'] = 0
    df.loc[df.Score >= 93, 'Score'] = 1
    #df['Score'] = np.where(df['Score'] >=93, 1, df['Score'])
    #df['Score'] = np.where(df['Score'] < 93, 0, df['Score'])
    #replace_map={df['Score']>=93:1,df['Score']<93:0}
    #df.replace(replace_map,inplace=True)
    return df

file_name = 'C:\develop\DAproject/WineQuality.csv'
raw_dataset = load_dataset(file_name)
cln_dataset = remove_missing_values(raw_dataset)
cln_dataset = remove_duplicate_rows(cln_dataset)
cln_dataset=remove_not_years(cln_dataset)
cln_dataset=remove_any_wine_not_whithorred(cln_dataset,'Category')
cln_dataset=clean_alcohol(cln_dataset)
replace_map={'White':2,'Red':1}
cln_dataset.replace(replace_map,inplace=True)
cln_dataset['Score']=cln_dataset['Score'].str.findall('\d+').str[0].astype('float')
cln_dataset=replace_score(cln_dataset)

cln_dataset.to_csv('C:\develop\DAproject/CleanWineQuality.csv')



