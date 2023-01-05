from bs4 import BeautifulSoup
import requests
import random
import pandas as pd
import re

user_agent = user_agents_list = [
    'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'
]
url1="https://www.winemag.com/?s=&drink_type=wine&page="
names=[]
price=[]
score=[]
alcohol=[]
year=[]
bottle=[]
category=[]
winery=[]
count=0
fromPlace=[]

for i in range(1,2):
     url2=url1+str(i)+"&search_type=reviews"
     response1 = requests.get(url2,headers={'User-Agent': random.choice(user_agents_list)})
     soup1 = BeautifulSoup(response1.content, "html.parser")
     infos=soup1.findAll("a",attrs={"class":"review-listing row"})
     for data in infos:
         names.append(data.find("h3",attrs={"class":"title"}).text)
         price.append(data.find("span",attrs={"class":"price"}).text)
         score.append(data.find("span", attrs={"class": "rating"}).text)
         url3=data['href']
         response2 = requests.get(url3, headers={'User-Agent': random.choice(user_agents_list)})
         soup2 = BeautifulSoup(response2.content, "html.parser")
         alcohol.append(soup2.find("div",attrs={"class":"info small-9 columns"}).text)
         bottle.append(soup2.findAll("div",attrs={"class":"info small-9 columns"})[1].text)
         category.append(soup2.findAll("div",attrs={"class":"info small-9 columns"})[2].text)
         fromPlace.append(soup2.findAll("div",attrs={"class":"info medium-9 columns"})[3].text)
         #winery.append(soup2.findAll("div",attrs={"class":"info medium-9 columns"})[4].text)

data={'Name':names,'Price':price,'Score':score,'Alcohol':alcohol,'Bottle':bottle,'Category':category,'From':fromPlace,'Winery':winery}
df = pd.DataFrame(data)
df['Year']=df['Name'].str.findall('\d+').str[0].astype('Int64')
df['Alcohol']=df['Alcohol'].str.findall('\d+').str[0].astype('float')
df['Bottle']=df['Bottle'].str.findall('\d+').str[0].astype('Int64')
print(df)

