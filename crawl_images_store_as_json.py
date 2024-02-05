import requests
import json
from bs4 import BeautifulSoup

from urllib.parse import urljoin

urls=["http://127.0.0.1:5000/a",
           "http://127.0.0.1:5000/b",
           "http://127.0.0.1:5000/c",
           "http://127.0.0.1:5000/d",
           "http://127.0.0.1:5000/e"]
 
images=[]

for url in urls:
    page=requests.get(url)
    soup=BeautifulSoup(page.content,'html.parser')
    
    img_tags=soup.find_all('img')
    for img in img_tags:
        if 'src' in img.attrs:
            img_url=urljoin(url, img.attrs['src'])
            alt_text=img.attrs.get('alt','')
            title=img.attrs.get('title','')
            img_data={'url':img_url,'alt_text': alt_text.lower(), 
            'title':title.lower(), 'source_url':url}
            
        if img_data not in images:
            images.append(img_data)

with open('images.json','w') as f:
    json.dump(images, f, indent=2)
    
    
    
            
        