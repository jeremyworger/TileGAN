from bs4 import BeautifulSoup
from urllib2 import urlopen
import urllib

# image scraper functions
def make_soup(url):
    html = urlopen(url).read()
    return BeautifulSoup(html, "html5lib")

def get_images(url):
    soup = make_soup(url)
    images = [img for img in soup.findAll('img')]
    print(str(len(images)) + "images found.")
    print("Downloading images to current working directory.")

    image_links = [each.get('src') for each in images]
    for each in image_links:
        filename = each.split('/')[-1]
        urllib.urlretrieve(each, filename)
    return image_links

if __name__ == "__main__":
    get_images("place URL here")