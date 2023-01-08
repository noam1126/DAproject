import pandas as pd
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


file_name = 'C:\develop\DAproject/WineQuality.csv'
raw_dataset = load_dataset(file_name)
cln_dataset = remove_missing_values(raw_dataset)
cln_dataset = remove_duplicate_rows(cln_dataset)
cln_dataset=remove_not_years(cln_dataset)
cln_dataset=remove_any_wine_not_whithorred(cln_dataset,'Category')

cln_dataset.to_csv('C:\develop\DAproject/CleanWineQuality.csv')



