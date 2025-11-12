# app.py
"""
Streamlit app to compare fact-check verdicts between
BOOM Live (India) and Alt News (India).

How to run:
    pip install -r requirements.txt
    streamlit run app.py

Suggested requirements.txt:
streamlit
requests
beautifulsoup4
pandas
rapidfuzz
matplotlib
scikit-learn
python-dateutil

"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from rapidfuzz import process, fuzz
from datetime import datetime
from dateutil import parser as dateparser
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import io

st.set_page_config(page_title="BOOM vs AltNews — Fact-check comparison", layout="wide")

# ---------------------------
# Utility / normalization
# ---------------------------

VERDICT_MAPPING = {
    # Generic mapping to numeric [0..1] and canonical label
    # Adjust or extend as you see fit
    'true': (1.0, 'True'),
    'correct': (1.0, 'True'),
    'mostly true': (0.9, 'Mostly True'),
    'partly true': (0.6, 'Partly True'),
    'half true': (0.5, 'Half True'),
    'mixture': (0.5, 'Mixture'),
    'mixed': (0.5, 'Mixture'),
    'uncertain': (0.5, 'Uncertain'),
    'no evidence': (0.2, 'No Evidence'),
    'mostly false': (0.1, 'Mostly False'),
    'false': (0.0, 'False'),
    'misleading': (0.2, 'Misleading'),
    'partly false': (0.2, 'Partly False'),
    'satire': (0.0, 'Satire'),
    'missing context': (0.3, 'Missing Context'),
    'no verdict': (0.5, 'No Verdict'),
}

def normalize_verdict(text):
    if not text or not isinstance(text, str):
        return 0.5, 'No Verdict'
    t = text.strip().lower()
    # try to find a key in mapping contained in t
    for k in VERDICT_MAPPING:
        if k in t:
            return VERDICT_MAPPING[k]
    # fallback heuristics
    if 'true' in t:
        return 1.0, 'True'
    if 'false' in t:
        return 0.0, 'False'
    if 'mislead' in t:
        return 0.2, 'Misleading'
    return 0.5, text.strip().title()

# ---------------------------
# Scrapers
# ---------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; fact-compare-bot/1.0; +https://example.com/bot)"
}

@st.cache_data(show_spinner=False)
def fetch_altnews_pages(pages=1):
    """
    Scrape Alt News fact-check listing pages.
    Returns DataFrame with columns: claim, verdict_raw, url, date
    """
    base = "https://www.altnews.in"
    listing_url = base + "/category/fact-checks/page/{page}/"
    rows = []
    for p in range(1, pages+1):
        url = listing_url.format(page=p)
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # AltNews lists posts in article tags or elements with "post"
            articles = soup.select("article, .post")
            if not articles:
                # try common backup
                articles = soup.select(".grid-item a")
            for a in articles:
                # find title and link
                link = a.find("a", href=True)
                if link:
                    post_url = link['href']
                else:
                    # fallback - find first link inside
                    link = a.select_one("a[href]")
                    post_url = link['href'] if link else None
                title = None
                title_tag = a.select_one("h2, h3, .entry-title, .post-title")
                if title_tag:
                    title = title_tag.get_text(strip=True)
                else:
                    # fallback: link text
                    title = link.get_text(strip=True) if link else None

                # Visit post page to extract verdict and date
                if post_url:
                    try:
                        pr = requests.get(post_url, headers=HEADERS, timeout=12)
                        pr.raise_for_status()
                        psoup = BeautifulSoup(pr.text, "html.parser")
                        # common altnews selectors: verdict sometimes in a highlighted box, or within h2/h3 with 'Verdict' word
                        verdict = None
                        # look for 'Verdict' heading then capture sibling text
                        vnode = psoup.find(lambda tag: tag.name in ['h2','h3','strong'] and 'verdict' in tag.get_text(strip=True).lower())
                        if vnode:
                            # get next sibling or next p
                            nxt = vnode.find_next_sibling(['p','div','span'])
                            if nxt:
                                verdict = nxt.get_text(" ", strip=True)
                        # try meta or badges
                        if not verdict:
                            # some posts have labels or spans with class 'rating' or 'verdict'
                            vtag = psoup.select_one(".verdict, .rating, .result, .fact-check-result")
                            if vtag:
                                verdict = vtag.get_text(" ", strip=True)
                        if not verdict:
                            # fallback: search for common words in body
                            body = psoup.get_text(" ", strip=True)
                            # attempt to find substring 'Verdict:' in body
                            if 'verdict' in body.lower():
                                idx = body.lower().find('verdict')
                                snippet = body[idx: idx+200]
                                verdict = snippet.split(':',1)[-1].strip()
                        # date
                        date = None
                        time_tag = psoup.find('time')
                        if time_tag and time_tag.has_attr('datetime'):
                            try:
                                date = dateparser.parse(time_tag['datetime'])
                            except:
                                pass
                        if not date:
                            # try meta
                            meta = psoup.find('meta', {'property':'article:published_time'}) or psoup.find('meta', {'name':'pubdate'})
                            if meta and meta.get('content'):
                                try:
                                    date = dateparser.parse(meta['content'])
                                except:
                                    date = None
                        rows.append({
                            'claim': title,
                            'verdict_raw': verdict,
                            'url': post_url,
                            'date': date
                        })
                    except Exception as e:
                        # skip problematic post, continue
                        continue
        except Exception as e:
            continue
    df = pd.DataFrame(rows)
    return df

@st.cache_data(show_spinner=False)
def fetch_boom_pages(pages=1):
    """
    Scrape BOOM Live fact-check listing pages.
    Returns DataFrame with columns: claim, verdict_raw, url, date
    """
    base = "https://www.boomlive.in"
    listing_url = base + "/fact-checks/page/{page}/"
    rows = []
    for p in range(1, pages+1):
        url = listing_url.format(page=p)
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            articles = soup.select("article, .post, .listing")
            for a in articles:
                link = a.find("a", href=True)
                if link:
                    post_url = link['href']
                else:
                    post_url = None
                title_tag = a.select_one("h2, h3, .entry-title, .post-title")
                title = title_tag.get_text(strip=True) if title_tag else (link.get_text(strip=True) if link else None)

                if post_url:
                    try:
                        pr = requests.get(post_url, headers=HEADERS, timeout=12)
                        pr.raise_for_status()
                        psoup = BeautifulSoup(pr.text, "html.parser")
                        verdict = None
                        vnode = psoup.find(lambda tag: tag.name in ['h2','h3','strong'] and 'verdict' in tag.get_text(strip=True).lower())
                        if vnode:
                            nxt = vnode.find_next_sibling(['p','div','span'])
                            if nxt:
                                verdict = nxt.get_text(" ", strip=True)
                        if not verdict:
                            vtag = psoup.select_one(".verdict, .result, .factcheck-result, .rating")
                            if vtag:
                                verdict = vtag.get_text(" ", strip=True)
                        if not verdict:
                            body = psoup.get_text(" ", strip=True)
                            if 'verdict' in body.lower():
                                idx = body.lower().find('verdict')
                                snippet = body[idx: idx+200]
                                verdict = snippet.split(':',1)[-1].strip()
                        # date
                        date = None
                        time_tag = psoup.find('time')
                        if time_tag and time_tag.has_attr('datetime'):
                            try:
                                date = dateparser.parse(time_tag['datetime'])
                            except:
                                pass
                        if not date:
                            meta = psoup.find('meta', {'property':'article:published_time'}) or psoup.find('meta', {'name':'pubdate'})
                            if meta and meta.get('content'):
                                try:
                                    date = dateparser.parse(meta['content'])
                                except:
                                    date = None
                        rows.append({
                            'claim': title,
                            'verdict_raw': verdict,
                            'url': post_url,
                            'date': date
                        })
                    except Exception as e:
                        continue
        except Exception as e:
            continue
    df = pd.DataFrame(rows)
    return df

# ---------------------------
# Matching and comparison
# ---------------------------

def fuzzy_match_claims(df_a, df_b, score_cutoff=80, limit=1):
    """
    For each claim in df_a, find best match in df_b using fuzzy matching.
    Returns merged DataFrame with pairings where match score >= score_cutoff.
    """
    choices = df_b['claim'].fillna('').tolist()
    mapping = []
    for idx, row in df_a.iterrows():
        claim = row['claim'] or ''
        if not claim.strip():
            continue
        best = process.extractOne(claim, choices, scorer=fuzz.token_sort_ratio)
        if best:
            match_text, score, pos = best  # best is (string, score, index)
            if score >= score_cutoff:
                matched_row = df_b.iloc[pos]
                mapping.append({
                    'claim_a': claim,
                    'verdict_a_raw': row.get('verdict_raw'),
                    'url_a': row.get('url'),
                    'date_a': row.get('date'),
                    'claim_b': matched_row.get('claim'),
                    'verdict_b_raw': matched_row.get('verdict_raw'),
                    'url_b': matched_row.get('url'),
                    'date_b': matched_row.get('date'),
                    'fuzzy_score': score
                })
    return pd.DataFrame(mapping)

def prepare_comparison_df(m):
    """
    Normalize verdicts and create numeric columns.
    """
    if m.empty:
        return m
    m = m.copy()
    m[['score_a','verdict_a_norm']] = m['verdict_a_raw'].apply(lambda x: pd.Series(normalize_verdict(x)))
    m[['score_b','verdict_b_norm']] = m['verdict_b_raw'].apply(lambda x: pd.Series(normalize_verdict(x)))
    # Add binary labels for confusion matrix (True vs False) using threshold 0.5
    m['binary_a'] = (m['score_a'] >= 0.5).astype(int)
    m['binary_b'] = (m['score_b'] >= 0.5).astype(int)
    return m

def compute_agreement_stats(df):
    if df.empty:
        return {}
    total = len(df)
    agree_exact_label = (df['verdict_a_norm'] == df['verdict_b_norm']).sum()
    agree_binary = (df['binary_a'] == df['binary_b']).sum()
    # correlation
    corr = df['score_a'].corr(df['score_b'])
    return {
        'total_pairs': total,
        'agree_exact_label': int(agree_exact_label),
        'agree_exact_pct': float(agree_exact_label) / total * 100,
        'agree_binary': int(agree_binary),
        'agree_binary_pct': float(agree_binary) / total * 100,
        'score_correlation': corr
    }

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("BOOM Live ↔ Alt News — Fact-check Comparison")
st.markdown(
    "Scrapes recent fact-checks from **BOOM Live** and **Alt News** (India), fuzzy-matches claims, "
    "normalizes verdict labels and compares truthfulness."
)

with st.sidebar:
    st.header("Scrape & Match Settings")
    pages_alt = st.number_input("Alt News pages to scrape", min_value=1, max_value=10, value=2)
    pages_boom = st.number_input("BOOM pages to scrape", min_value=1, max_value=10, value=2)
    fuzzy_cutoff = st.slider("Fuzzy match cutoff (0-100)", min_value=50, max_value=100, value=85)
    show_examples = st.checkbox("Show matched examples", value=True)
    st.markdown("---")
    st.write("Options")
    refresh = st.button("Refresh / Re-scrape (clears cache)")

if refresh:
    # Clear cached data functions by calling their cache_clear
    try:
        fetch_altnews_pages.clear()
        fetch_boom_pages.clear()
        st.experimental_rerun()
    except Exception:
        st.warning("Could not clear cache programmatically. Reloading page should fetch fresh data.")

st.info("Scraping pages now — this may take a few seconds depending on pages requested. Results are cached.")

with st.spinner("Fetching Alt News..."):
    df_alt = fetch_altnews_pages(pages=pages_alt)
with st.spinner("Fetching BOOM Live..."):
    df_boom = fetch_boom_pages(pages=pages_boom)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Alt News")
    st.write(f"Scraped posts: {len(df_alt)}")
    st.dataframe(df_alt[['claim','verdict_raw','date','url']].head(50))
with col2:
    st.subheader("BOOM Live")
    st.write(f"Scraped posts: {len(df_boom)}")
    st.dataframe(df_boom[['claim','verdict_raw','date','url']].head(50))

# Allow user to optionally upload CSVs instead of scraping
st.markdown("---")
st.subheader("Optional: Upload your own CSVs instead of scraping")
st.markdown("CSV should have columns: `claim`, `verdict_raw`, `url` (optional), `date` (optional)")
upload_alt = st.file_uploader("Upload Alt News CSV", type=['csv'], key='u1')
upload_boom = st.file_uploader("Upload BOOM CSV", type=['csv'], key='u2')

if upload_alt is not None:
    try:
        user_alt = pd.read_csv(upload_alt)
        df_alt = user_alt
        st.success("Loaded Alt CSV - using uploaded data.")
    except Exception as e:
        st.error("Failed to read Alt CSV: " + str(e))
if upload_boom is not None:
    try:
        user_boom = pd.read_csv(upload_boom)
        df_boom = user_boom
        st.success("Loaded BOOM CSV - using uploaded data.")
    except Exception as e:
        st.error("Failed to read BOOM CSV: " + str(e))

# Run fuzzy matching in both directions optionally, but primary: Alt -> Boom
st.markdown("---")
st.subheader("Matching & Comparison")
if df_alt.empty or df_boom.empty:
    st.warning("Need non-empty data from both sources. Try increasing pages to scrape or upload CSVs.")
else:
    with st.spinner("Fuzzy-matching claims..."):
        matches = fuzzy_match_claims(df_alt, df_boom, score_cutoff=fuzzy_cutoff)
        comp = prepare_comparison_df(matches)
    stats = compute_agreement_stats(comp)
    st.metric("Matched pairs", stats.get('total_pairs', 0))
    st.metric("Exact-label agreement (%)", f"{stats.get('agree_exact_pct', 0):.1f}%")
    st.metric("Binary agreement (%)", f"{stats.get('agree_binary_pct', 0):.1f}%")
    corr = stats.get('score_correlation', None)
    st.metric("Score correlation (Pearson)", f"{corr:.3f}" if pd.notna(corr) else "N/A")

    if comp.empty:
        st.info("No matches above the fuzzy cutoff. Try lowering the cutoff or scraping more pages.")
    else:
        if show_examples:
            st.subheader("Matched examples (top 20 by fuzzy score)")
            st.dataframe(comp.sort_values('fuzzy_score', ascending=False).head(20)[[
                'fuzzy_score','claim_a','verdict_a_raw','verdict_a_norm','score_a',
                'claim_b','verdict_b_raw','verdict_b_norm','score_b','url_a','url_b'
            ]])

        # Scatter plot of numeric scores
        st.subheader("Numeric Verdict Comparison (normalized scores)")
        fig, ax = plt.subplots()
        ax.scatter(comp['score_a'], comp['score_b'])
        ax.set_xlabel("Alt News normalized score")
        ax.set_ylabel("BOOM normalized score")
        ax.set_title("Scatter: Alt News vs BOOM normalized verdict scores")
        # draw y=x line
        ax.plot([0,1],[0,1], linestyle='--')
        st.pyplot(fig)

        # Confusion matrix for binary labels
        st.subheader("Binary Confusion Matrix (True vs False)")
        y_true = comp['binary_a']
        y_pred = comp['binary_b']
        cm = confusion_matrix(y_true, y_pred, labels=[1,0])  # 1=True, 0=False
        fig2, ax2 = plt.subplots()
        im = ax2.imshow(cm, interpolation='nearest')
        ax2.set_xlabel('BOOM predicted (binary)')
        ax2.set_ylabel('Alt News actual (binary)')
        ax2.set_xticks([0,1])
        ax2.set_xticklabels(['True','False'])
        ax2.set_yticks([0,1])
        ax2.set_yticklabels(['True','False'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="w")
        ax2.set_title("Confusion matrix (AltNews vs BOOM)")
        st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Export matched comparison")
        csv = comp.to_csv(index=False)
        st.download_button("Download matched CSV", data=csv, file_name="alt_boom_matched.csv", mime="text/csv")

st.markdown("---")
st.write("Developer notes:")
st.write("""
- If verdict extraction misses some pages, open the post URL and inspect HTML to adjust selectors in the scraping functions.
- You can change the `VERDICT_MAPPING` dictionary to better reflect site-specific labels.
- Fuzzy matching can be run in the other direction (BOOM -> Alt) to find additional pairs — you can add that similarly.
""")
st.write("Done — try adjusting pages and fuzzy cutoff to get more or fewer matches.")
