import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_snopes_articles(limit=15):
    url = "https://www.snopes.com/fact-check/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    data = []
    for item in soup.select(".media-list .media")[:limit]:
        title_elem = item.select_one(".media-heading a")
        rating_elem = item.select_one(".media-rating")
        if title_elem:
            title = title_elem.text.strip()
            rating = rating_elem.text.strip() if rating_elem else "Unknown"
            data.append({"source": "Snopes", "claim": title, "rating": rating})
    return pd.DataFrame(data)


def get_politifact_articles(limit=15):
    url = "https://www.politifact.com/factchecks/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    data = []
    for item in soup.select(".m-statement__content")[:limit]:
        title_elem = item.select_one(".m-statement__quote")
        rating_elem = item.select_one(".m-statement__meter img")
        if title_elem and rating_elem:
            title = title_elem.text.strip()
            rating = rating_elem["alt"].replace("Truth-O-Meter rating:", "").strip()
            data.append({"source": "PolitiFact", "claim": title, "rating": rating})
    return pd.DataFrame(data)
