# from bs4 import BeautifulSoup
# from urllib2 import urlopen
# import urllib

# # image scraper functions
# def make_soup(url):
#     html = urlopen(url).read()
#     return BeautifulSoup(html, "html5lib")

# def get_images(url):
#     # soup = make_soup(url)
#     uls = []
#     lis = [li for ul in uls for li in ul.findAll('li')]
#     print(lis)
#     # images = [img for img in soup.findAll('img')]
#     # print(str(len(images)) + "images found.")
#     # print("Downloading images to current working directory.")

#     # image_links = [each.get('src') for each in images]
#     # for each in image_links:
#     #     filename = each.split('/')[-1]
#     #     urllib.urlretrieve(each, filename)
#     # return image_links

import urllib2
from bs4 import BeautifulSoup

url = "http://www.vgmuseum.com/nes_b.html"
page = urllib2.urlopen(url).read()
soup = BeautifulSoup(page, "html5lib")

# filters links routing back to the same page
links = [link for link in soup.select('ol > li > a') if '_self' not in link.get('target', '')]

def read_link(new_page):
    game_page = urllib2.urlopen(new_page).read()
    game_soup = BeautifulSoup(game_page, "html5lib")
    images = [img for img in game_soup.findAll('img')]
    print(str(len(images)) + "images found.")

for link in links:
    read_link("http://www.vgmuseum.com/" + link.get('href'))
