from bs4 import BeautifulSoup
import requests
import random
import pandas as pd
import os

# At the begging we did crawling to our source website.
# It run on 370 pages so we will have enough variety of wines to our dataframe.
# Then run on every wine's page and crawled the relevant parameters.
# Every parameter we put into a col of the dataframe according to the list he belong.
# Our crawling is divided to 2 parts:
#   1.	From every wine list page we took the following parameters- name, price and score.
#   2.	From every specific wine page we took the following parameters- from where, category, bottle size, alcohol, winery and           variety.


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
variety=[]
count=0
fromPlace=[]

for i in range(1,350):
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
         bottle.append(soup2.findAll("div",attrs={"class":"info small-9 columns"})[1].text.replace("\n", ""))
         category.append(soup2.findAll("div",attrs={"class":"info small-9 columns"})[2].text.replace("\n", ""))
         fromPlace.append(data.find("span", attrs = {"class":"appellation"}).text.replace("\n", ""))
         for j in range(0,4):
             check=soup2.findAll("div",attrs={"class":"info-label medium-7 columns"})[j].text
             if(check=="\nVariety\n"):
                 variety.append(soup2.findAll("div",attrs={"class":"info medium-9 columns"})[j-1].text.replace("\n", ""))
                 break
         for j in range(0,7):
             check=soup2.findAll("div",attrs={"class":"info-label medium-7 columns"})[j].text
             if(check=="\nWinery\n"):
                 winery.append(soup2.findAll("div",attrs={"class":"info medium-9 columns"})[j-1].text.replace("\n", ""))
                 break



data={'Name':names,'Price':price,'Alcohol':alcohol,'Bottle':bottle,'Category':category,'From':fromPlace,'Variety':variety,'Winery':winery,'Score':score}

df = pd.DataFrame(data)
df['Year']=df['Name'].str.findall('\d+').str[0].astype('Int64')
df['Alcohol']=df['Alcohol'].str.findall('\d+').str[0].astype('float')
df['Bottle']=df['Bottle'].str.findall('\d+').str[0].astype('Int64')

os.makedirs('C:\develop\DAproject', exist_ok=True)
df.to_csv('C:\develop\DAproject/WineQuality.csv')


