import urllib
import urllib2
from bs4 import BeautifulSoup

url = "http://www.vgmuseum.com/nes_b.html"
page = urllib2.urlopen(url).read()
soup = BeautifulSoup(page, "html5lib")
links = [link for link in soup.select('ol > li > a') if '_self' not in link.get('target', '')]

def get_backslash_count(s):
    """Returns the number of backslashes in a url"""
    return sum(1 for i in range(len(s)) if s[i] is '/')


def read_link(new_page):
    """Creates a new soup for each link in the games list"""
    game_page = urllib2.urlopen(new_page).read()
    game_soup = BeautifulSoup(game_page, "html5lib")
    images = game_soup.find_all('img', src=True)
    print(str(len(images)) + " images found.")

    for image in images:
        image = image["src"].split("src=")[-1]
        n = get_backslash_count(str(image))
        if n > 0:
            continue
        image_url = new_page.replace(new_page.split('/')[-1], image)
        urllib.urlretrieve(image_url, image)

for link in links:
    img_url = "http://www.vgmuseum.com/" + link.get('href')
    read_link(img_url)