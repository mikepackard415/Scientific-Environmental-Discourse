import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
referer = 'https://www.resilience.org/latest-articles/'
headers = {'User-Agent': user_agent, 'referer':referer}

emag = pd.read_pickle('../Data/Emagazine/emag.pkl')

link_text = {}
t0 = time.time()
for i in range(7915):
    
    if i % 200 == 0:
        print(i, (time.time() - t0) / 60)

    link = emag.link.iloc[i]
    response = requests.get(link, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    text = " ".join([p.text for p in soup.find('div', {'class':'entry-content'}).find_all('p')])
    text = text.replace('\xa0', '')
    
    lt = {'link':link, 'text':text}
    link_text[i] = lt

link_texts = pd.DataFrame({'link':[link_text[i]['link'] for i in range(7915)],
                           'text':[link_text[i]['text'] for i in range(7915)]})

try:
    link_texts.to_pickle('../Data/Emagazine/emag4.pkl')
except:
    link_texts.to_pickle('emag4.pkl')