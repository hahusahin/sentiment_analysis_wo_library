# IMPORT LIBRARIES
import requests
from bs4 import BeautifulSoup
import pandas as pd

# CONSTANTS

# Used to avoid detection as a bot
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", 
			"Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", 
			"DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}

# The url
base_url = ("https://www.amazon.com.tr/Xiaomi-Wireless-Bluetooth-Kablosuz-Kulakl%C4%B1k/product-reviews/B08GSQ7B9Z/"
				"ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={}")

# Number of pages
no_pages = 30 


# WEB-SCRAPING
results = []

for page in range(1, no_pages+1):
	res = requests.get(base_url.format(page), headers=headers)
	soup = BeautifulSoup(res.text, 'lxml')

	collections = soup.find_all('span', attrs={'class':'a-size-base review-text review-text-content'})
	for item in collections:
		review = item.select('span > span')[0].text.strip()
		if len(review) < 300:
			results.append(review)

# WRITE TO DISK
df = pd.DataFrame(results, columns=['YORUM'])
df.to_excel('yorumlar.xlsx', index=False)
