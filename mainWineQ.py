from bs4 import BeautifulSoup
import requests
import random

user_agent = user_agents_list = [
    'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'
]
url1="https://www.winemag.com/?s=&drink_type=wine&page="

count=0
for i in range(1,250):
     url2=url1+str(i)+"&search_type=reviews"
     response1 = requests.get(url2,headers={'User-Agent': random.choice(user_agents_list)})
     soup1 = BeautifulSoup(response1.content, "html.parser")
     for t in soup1.findAll("div",attrs={"class":"hide-for-small"}):
          count=count+1
print(count)
