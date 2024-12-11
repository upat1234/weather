# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# This is a class that can download a link and get all links within a page.
class Crawler:
    def __init__(self):
        self.urls = []
    
    # Get the text from a url
    def download_url(self, url):
        return requests.get(url).text
    
    # Get the links from the text of a url.
    def get_links(self, url, text):
        self.urls = []
        soup = BeautifulSoup(text, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path != '../':
                path = urljoin(url, path)
                self.urls.append(path)
        return self.urls
