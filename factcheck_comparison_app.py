import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser

st.set_page_config(page_title="Fact Check Comparison", layout="wide")

st.title("📰 News Truthfulness Comparison App")
st.markdown("""
This app compares fact-checks from **BOOM Live (India)** and **Alt News (India)**.
If direct scraping fails, it automatically falls back to their RSS feeds.
""")

# --------------------------------------------------------
# HTML Scrapers (primary sources)
# --------------------------------------------------------

def fetch_altnews_pages(pages=1):
    base_url = "https://www.altnews.in/page/"
    all_data = []

    for page in range(1, pages + 1):
        url = f"{base_url}{page}/"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code != 200:
            continue
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("article")

        for art in articles:
            title = art.find("h2")
            link = title.find("a")["href"] if title and title.find("a") else None
            claim = title.get_text(strip=True) if title else None
            date_tag = art.find("time")
            date = date_tag.get("datetime") if date_tag else None
            verdict_raw = None
            if art.find("div", class_="entry-content"):
                verdict_raw = art.find("div", class_="entry-content").get_text(strip=True)[:150]
            all_data.append({
                "claim": claim,
                "verdict_raw": verdict_raw,
                "date": date,
                "url": link
            })

    return pd.DataFrame(all_data)

def fetch_boom_pages(pages=1):
    base_url = "https://www.boomlive.in/fact-check"
    all_data = []

    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code != 200:
            continue
        soup = BeautifulSoup(res.text, "html.parser")
        cards = soup.find_all("a", class_="story-card")

        for c in cards:
            title = c.find("h2")
            claim = title.get_text(strip=True) if title else None
            link = "https://www.boomlive.in" + c["href"] if c.get("href") else None
            verdict_raw = None
            desc = c.find("p")
            if desc:
                verdict_raw = desc.get_text(strip=True)[:150]
            all_data.append({
                "claim": claim,
                "verdict_raw": verdict_raw,
                "date": None,
                "url": link
            })

    return pd.DataFrame(all_data)

# --------------------------------------------------------
# RSS Fallbacks
# --------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_altnews_rss(limit=20):
    url = "https://www.altnews.in/feed/"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:limit]:
        rows.append({
            "claim": entry.title,
            "verdict_raw": None,
            "url": entry.link,
            "date": entry.published if "published" in entry else None
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def fetch_boom_rss(limit=20):
    url = "https://www.boomlive.in/feed"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:limit]:
        rows.append({
            "claim": entry.title,
            "verdict_raw": None,
            "url": entry.link,
            "date": entry.published if "published" in entry else None
        })
    return pd.DataFrame(rows)

# --------------------------------------------------------
# Combined fetch logic (with fallback)
# --------------------------------------------------------

def get_data_with_fallback(pages, fetch_html_func, fetch_rss_func, label):
    try:
        df = fetch_html_func(pages)
        if df is None or df.empty or "claim" not in df.columns:
            st.warning(f"⚠️ {label} HTML scraping returned no data — using RSS fallback.")
            df = fetch_rss_func(limit=20)
        return df
    except Exception as e:
        st.warning(f"⚠️ {label} scraping failed: {e}. Using RSS fallback instead.")
        return fetch_rss_func(limit=20)

# --------------------------------------------------------
# Display helper
# --------------------------------------------------------

def safe_show(df, label):
    st.subheader(label)
    if df.empty:
        st.warning(f"No data scraped from {label}. Try increasing pages or refreshing.")
        return
    st.write(f"✅ Scraped posts: {len(df)}")
    cols = [c for c in ['claim', 'verdict_raw', 'date', 'url'] if c in df.columns]
    if cols:
        st.dataframe(df[cols].head(50))
    else:
        st.error(f"{label} data missing expected columns. Found: {list(df.columns)}")

# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------

st.sidebar.header("Settings")
pages_alt = st.sidebar.slider("Alt News Pages to Fetch", 1, 5, 1)
pages_boom = st.sidebar.slider("BOOM Live Pages to Fetch", 1, 5, 1)

with st.spinner("Fetching data..."):
    df_alt = get_data_with_fallback(pages_alt, fetch_altnews_pages, fetch_altnews_rss, "Alt News")
    df_boom = get_data_with_fallback(pages_boom, fetch_boom_pages, fetch_boom_rss, "BOOM Live")

# --------------------------------------------------------
# Display results
# --------------------------------------------------------

st.header("🧾 Scraped Fact-Checks")
col1, col2 = st.columns(2)

with col1:
    safe_show(df_alt, "Alt News")
with col2:
    safe_show(df_boom, "BOOM Live")

# --------------------------------------------------------
# Search functionality
# --------------------------------------------------------

st.header("🔍 Compare News Truthfulness")
query = st.text_input("Enter a news headline or claim:")

if query:
    def search_match(df, query):
        if "claim" not in df.columns:
            return pd.DataFrame()
        return df[df["claim"].str.contains(query, case=False, na=False)]

    results_alt = search_match(df_alt, query)
    results_boom = search_match(df_boom, query)

    st.subheader("🟦 Alt News Results")
    if not results_alt.empty:
        st.dataframe(results_alt[["claim", "verdict_raw", "url"]].head(10))
    else:
        st.info("No matching results on Alt News.")

    st.subheader("🟨 BOOM Live Results")
    if not results_boom.empty:
        st.dataframe(results_boom[["claim", "verdict_raw", "url"]].head(10))
    else:
        st.info("No matching results on BOOM Live.")

    if not results_alt.empty and not results_boom.empty:
        st.success("✅ Both sites have covered this topic — cross-check recommended!")

st.markdown("---")
st.caption("Developed for comparing verified fact-checks across major Indian sources.")
