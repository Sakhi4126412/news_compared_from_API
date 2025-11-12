# app.py
"""
Streamlit app to compare truthfulness judgments between PolitiFact and FactCheck.org.

Save as app.py and run:
    streamlit run app.py

Notes:
- Scrapes both sites; HTML structures can change over time and break the parsers.
- For more stable production use, prefer official APIs or RSS feeds (if available).
- Be polite: don't hammer the servers. This app uses streamlit cache to reduce requests.
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from rapidfuzz import fuzz, process
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="PolitiFact ↔ FactCheck.org Truth Comparison", layout="wide")

# --------------------------
# Utilities & Normalization
# --------------------------
def normalize_rating(site: str, raw: str) -> str:
    """Map raw rating string to canonical labels: TRUE, MOSTLY_TRUE, MIXED, MOSTLY_FALSE, FALSE, PANTS_ON_FIRE, UNKNOWN"""
    if raw is None:
        return "UNKNOWN"
    text = raw.strip().lower()
    # PolitiFact specific:
    if "pants on fire" in text:
        return "PANTS_ON_FIRE"
    if "false" == text or text == "false." or "false" in text and "mostly" not in text:
        # handle PolitiFact "False" and also FactCheck "False"
        if "mostly" in text:
            return "MOSTLY_FALSE"
        return "FALSE"
    if "mostly false" in text:
        return "MOSTLY_FALSE"
    if "mostly true" in text:
        return "MOSTLY_TRUE"
    if "true" == text or text == "true." or ("true" in text and "mostly" not in text):
        return "TRUE"
    if "half true" in text or "mixed" in text or "mixture" in text:
        return "MIXED"
    if "misleading" in text or "mostly misleading" in text:
        return "MOSTLY_FALSE"
    # FactCheck.org sometimes uses phrases rather than labels:
    if "true" in text:
        return "TRUE"
    if "false" in text:
        return "FALSE"
    # fallback:
    return "UNKNOWN"

def rating_to_numeric(r: str) -> float:
    """Convert canonical rating to numeric scale 0.0 (False) .. 1.0 (True)."""
    mapping = {
        "PANTS_ON_FIRE": 0.0,
        "FALSE": 0.0,
        "MOSTLY_FALSE": 0.15,
        "MIXED": 0.5,
        "MOSTLY_TRUE": 0.85,
        "TRUE": 1.0,
        "UNKNOWN": np.nan
    }
    return mapping.get(r, np.nan)

def safe_get(url, headers=None, timeout=12):
    headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; PolitiFact-FactCheckComparer/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

# --------------------------
# Scrapers (simple)
# --------------------------
@st.cache_data(show_spinner=False)
def scrape_politifact_pages(num_pages=2):
    """
    Scrapes PolitiFact 'factchecks' index pages.
    Returns: DataFrame with columns: claim, rating_raw, url, date
    """
    base = "https://www.politifact.com/factchecks/list/?page={}"
    rows = []
    for p in range(1, num_pages+1):
        html = safe_get(base.format(p))
        soup = BeautifulSoup(html, "html.parser")
        # PolitiFact list items
        items = soup.select("li.o-listicle__item") or soup.select("div.m-list__item")
        if not items:
            items = soup.select(".m-statement")  # fallback
        for it in items:
            # claim
            claim_tag = it.select_one(".m-statement__quote") or it.select_one(".statement__quote") or it.select_one(".m-statement__content a")
            claim = claim_tag.get_text(strip=True) if claim_tag else None
            # rating
            rating_tag = it.select_one(".m-statement__meter .c-image") or it.select_one(".m-statement__meter img") or it.select_one(".rating")
            rating = None
            if rating_tag:
                # PolitiFact often uses alt text or title attribute
                rating = rating_tag.get("alt") or rating_tag.get("title") or rating_tag.get_text(strip=True)
            # link and date
            link_tag = it.select_one("a[href*='/factchecks/']") or it.select_one("a")
            url = "https://www.politifact.com" + link_tag["href"] if link_tag and link_tag.get("href", "").startswith("/") else (link_tag["href"] if link_tag else None)
            date_tag = it.select_one(".m-statement__meta time") or it.select_one("time")
            date = None
            if date_tag:
                date = date_tag.get("datetime") or date_tag.get_text(strip=True)
            rows.append({"claim": claim, "rating_raw": rating, "source": "PolitiFact", "url": url, "date": date})
    df = pd.DataFrame(rows)
    # normalize date string
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['rating_norm'] = df['rating_raw'].apply(lambda x: normalize_rating("politifact", x))
    df['rating_numeric'] = df['rating_norm'].apply(rating_to_numeric)
    return df

@st.cache_data(show_spinner=False)
def scrape_factcheck_pages(num_pages=2):
    """
    Scrapes FactCheck.org pages tagged 'FactChecking' or the site homepage list.
    Returns DataFrame with columns: claim, rating_raw, url, date
    """
    # FactCheck.org has a 'Fact-check' tag listing - we'll use pages like: https://www.factcheck.org/category/fact-check/
    base = "https://www.factcheck.org/category/fact-check/page/{}/"
    rows = []
    for p in range(1, num_pages+1):
        html = safe_get(base.format(p))
        soup = BeautifulSoup(html, "html.parser")
        posts = soup.select("article") or soup.select(".post")
        for post in posts:
            title_tag = post.select_one(".entry-title a") or post.select_one("h2 a") or post.select_one("h1 a")
            title = title_tag.get_text(strip=True) if title_tag else None
            url = title_tag["href"] if title_tag and title_tag.get("href") else None
            # We'll open the post page to find the claim text and any verdict text inside the content
            claim_text = title
            rating_text = None
            date_tag = post.select_one("time") or post.select_one(".entry-date")
            date = date_tag.get("datetime") if date_tag and date_tag.get("datetime") else (date_tag.get_text(strip=True) if date_tag else None)
            # Visit article page to search for likely verdict language
            if url:
                try:
                    art = safe_get(url)
                    asoup = BeautifulSoup(art, "html.parser")
                    # FactCheck.org often includes a summary sentence near the top like "Verdict: False" or "Our ruling: False"
                    content_text = asoup.select_one(".entry-content") or asoup.select_one(".post-content") or asoup
                    content = content_text.get_text(" ", strip=True)[:800] if content_text else ""
                    # find occurrence of "Verdict", "Ruling", "False", "True", "Misleading" etc.
                    for token in ["Verdict:", "Verdict –", "Our ruling:", "Ruling:", "Bottom line:", "Conclusion:", "Rating:"]:
                        if token in content:
                            # take a small slice after token
                            idx = content.find(token)
                            snippet = content[idx: idx + 120]
                            rating_text = snippet
                            break
                    # last resort: search content for the words true/false/misleading
                    if not rating_text:
                        for word in ["False", "True", "Mostly false", "Mostly true", "Misleading", "Pants on fire", "Mixture"]:
                            if word.lower() in content.lower():
                                rating_text = word
                                break
                    # sometimes the article subtitle contains summary
                    subtitle = asoup.select_one(".entry-subtitle") or asoup.select_one(".subtitle")
                    if not rating_text and subtitle:
                        rating_text = subtitle.get_text(strip=True)
                except Exception as e:
                    rating_text = None
            rows.append({"claim": claim_text, "rating_raw": rating_text, "source": "FactCheck.org", "url": url, "date": date})
    df = pd.DataFrame(rows)
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['rating_norm'] = df['rating_raw'].apply(lambda x: normalize_rating("factcheck", x))
    df['rating_numeric'] = df['rating_norm'].apply(rating_to_numeric)
    return df

# --------------------------
# Matching claims across sites
# --------------------------
def fuzzy_merge(df_left, df_right, left_on='claim', right_on='claim', threshold=80, limit=1):
    """
    Fuzzy match two dataframes on text columns.
    Returns merged DataFrame with left claim and best matching right claim when score >= threshold.
    """
    right_choices = df_right[right_on].astype(str).tolist()
    matches = []
    for i, left_val in df_left[left_on].astype(str).iteritems():
        if not left_val or left_val.strip() == "":
            matches.append((None, 0, None))
            continue
        best = process.extractOne(left_val, right_choices, scorer=fuzz.token_sort_ratio)
        if best:
            match_str, score, idx = best  # idx is index in choices list
            if score >= threshold:
                right_row = df_right.iloc[idx]
                matches.append((right_row[right_on], score, right_row.name))
            else:
                matches.append((None, score, None))
        else:
            matches.append((None, 0, None))
    df_left = df_left.copy().reset_index(drop=True)
    df_left['matched_claim_right'] = [m[0] for m in matches]
    df_left['match_score'] = [m[1] for m in matches]
    df_left['matched_right_idx'] = [m[2] for m in matches]
    # attach right columns where matched
    right_subset = df_right.reset_index()
    merged_rows = []
    for i, r in df_left.iterrows():
        mr_idx = r['matched_right_idx']
        if pd.notna(mr_idx):
            right_row = right_subset[right_subset['index'] == mr_idx].squeeze()
            merged_rows.append(pd.concat([r, right_row.add_prefix('right_')]))
        else:
            merged_rows.append(r)
    merged = pd.DataFrame(merged_rows)
    return merged

# --------------------------
# Streamlit UI
# --------------------------
st.title("🔎 Compare Truthfulness: PolitiFact ↔ FactCheck.org")
st.markdown(
    """
    Scrape, normalize, and compare fact-check ratings from PolitiFact and FactCheck.org.
    - Pick how many pages to scrape (more pages = longer run time).
    - Fuzzy-match claims across the two sites to find pairs to compare.
    """
)

with st.sidebar:
    st.header("Scrape options")
    politifact_pages = st.number_input("PolitiFact pages to scrape", min_value=1, max_value=8, value=2, step=1)
    factcheck_pages = st.number_input("FactCheck.org pages to scrape", min_value=1, max_value=8, value=2, step=1)
    fuzzy_threshold = st.slider("Fuzzy match threshold (0-100)", min_value=50, max_value=100, value=82)
    run_scrape = st.button("Scrape & Compare")

if run_scrape:
    with st.spinner("Scraping PolitiFact..."):
        try:
            df_p = scrape_politifact_pages(int(politifact_pages))
            st.success(f"PolitiFact: {len(df_p)} items scraped.")
        except Exception as e:
            st.error(f"Error scraping PolitiFact: {e}")
            st.stop()
    with st.spinner("Scraping FactCheck.org..."):
        try:
            df_f = scrape_factcheck_pages(int(factcheck_pages))
            st.success(f"FactCheck.org: {len(df_f)} items scraped.")
        except Exception as e:
            st.error(f"Error scraping FactCheck.org: {e}")
            st.stop()

    # Show raw tables
    st.subheader("Raw scraped samples")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PolitiFact (sample)**")
        st.dataframe(df_p[['claim','rating_raw','rating_norm','rating_numeric','date_parsed','url']].head(10))
    with c2:
        st.markdown("**FactCheck.org (sample)**")
        st.dataframe(df_f[['claim','rating_raw','rating_norm','rating_numeric','date_parsed','url']].head(10))

    # Merge using fuzzy matching
    st.subheader("Fuzzy-match claims across sites")
    merged = fuzzy_merge(df_p, df_f, left_on='claim', right_on='claim', threshold=fuzzy_threshold)
    # Create comparison columns
    # For matched rows attach right-side ratings if present
    def get_right_rating(row):
        if 'right_rating_norm' in row.index:
            return row.get('right_rating_norm', None), row.get('right_rating_numeric', np.nan)
        return None, np.nan

    merged['right_rating_norm'] = merged.get('right_rating_norm', merged.get('rating_norm_right', None))
    merged['right_rating_numeric'] = merged.get('right_rating_numeric', merged.get('rating_numeric_right', np.nan))

    # Normalize guarantee: ensure columns exist
    if 'rating_norm' not in merged.columns:
        merged['rating_norm'] = merged['rating_norm']
    if 'rating_numeric' not in merged.columns:
        merged['rating_numeric'] = merged['rating_numeric']

    # Filter to matched only
    matched_pairs = merged[merged['matched_claim_right'].notna()].copy()
    st.write(f"Found **{len(matched_pairs)}** matched claim pairs (threshold={fuzzy_threshold}).")
    if matched_pairs.empty:
        st.info("No matched claims found with current threshold. Try lowering the threshold or increasing pages.")
    else:
        # Prepare for analysis
        matched_pairs['pf_rating_numeric'] = matched_pairs['rating_numeric'].astype(float)
        # attempt to fetch right rating numeric from right_ prefix if exists, else from columns
        if 'right_rating_numeric' in matched_pairs.columns and matched_pairs['right_rating_numeric'].notna().any():
            matched_pairs['fc_rating_numeric'] = matched_pairs['right_rating_numeric'].astype(float)
            matched_pairs['fc_rating_norm'] = matched_pairs['right_rating_norm']
        else:
            # maybe available as rating_numeric_right
            if 'rating_numeric_right' in matched_pairs.columns:
                matched_pairs['fc_rating_numeric'] = matched_pairs['rating_numeric_right'].astype(float)
                matched_pairs['fc_rating_norm'] = matched_pairs['rating_norm_right']
            else:
                matched_pairs['fc_rating_numeric'] = matched_pairs['rating_numeric'] * np.nan
                matched_pairs['fc_rating_norm'] = None

        # show matched table
        st.subheader("Matched pairs (sample)")
        display_cols = ['claim', 'rating_norm', 'pf_rating_numeric', 'matched_claim_right', 'fc_rating_norm', 'fc_rating_numeric', 'match_score', 'url', 'right_url'] if 'right_url' in matched_pairs.columns else ['claim','rating_norm','pf_rating_numeric','matched_claim_right','fc_rating_norm','fc_rating_numeric','match_score','url']
        # ensure columns exist
        for c in ['right_url','url','matched_claim_right']:
            if c not in matched_pairs.columns:
                matched_pairs[c] = None
        st.dataframe(matched_pairs[['claim','rating_norm','pf_rating_numeric','matched_claim_right','fc_rating_norm','fc_rating_numeric','match_score','url','right_url']].head(20))

        # Drop NaNs for numeric comparison
        comp = matched_pairs.dropna(subset=['pf_rating_numeric','fc_rating_numeric']).copy()
        if comp.empty:
            st.warning("No matched pairs have numeric ratings on both sides. The parsers may have failed to extract rating text for FactCheck.org. Consider increasing pages or inspecting the raw data.")
        else:
            # Correlation
            corr = comp['pf_rating_numeric'].corr(comp['fc_rating_numeric'], method='pearson')
            st.metric("Pearson correlation between numeric ratings", f"{corr:.3f}")

            # Scatter plot
            st.subheader("Scatter: PolitiFact vs FactCheck.org (numeric ratings)")
            fig = px.scatter(comp, x='pf_rating_numeric', y='fc_rating_numeric',
                             hover_data=['claim','matched_claim_right','match_score'],
                             labels={'pf_rating_numeric':'PolitiFact (numeric)','fc_rating_numeric':'FactCheck.org (numeric)'},
                             title="Rating scatter plot")
            st.plotly_chart(fig, use_container_width=True)

            # Binarize for agreement: define truth threshold (>=0.5 -> True)
            comp['pf_bin'] = (comp['pf_rating_numeric'] >= 0.5).astype(int)
            comp['fc_bin'] = (comp['fc_rating_numeric'] >= 0.5).astype(int)
            agreement = (comp['pf_bin'] == comp['fc_bin']).mean()
            st.write(f"Agreement rate (binary): **{agreement:.2%}** (threshold 0.5)")

            # Confusion matrix
            cm = confusion_matrix(comp['pf_bin'], comp['fc_bin'], labels=[1,0])
            fig2, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest')
            ax.set_title("Confusion matrix (rows: PolitiFact, cols: FactCheck.org) 1=True,0=False")
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            ax.set_xticklabels(['True','False'])
            ax.set_yticklabels(['True','False'])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="w")
            st.pyplot(fig2)

            # Show examples of disagreements
            st.subheader("Examples of disagreement (sample)")
            disag = comp[comp['pf_bin'] != comp['fc_bin']].copy()
            st.write(f"{len(disag)} disagreements found")
            if not disag.empty:
                # show a few with links
                sample = disag.sample(min(10, len(disag)), random_state=1)
                out = sample[['claim','rating_norm','pf_rating_numeric','matched_claim_right','fc_rating_norm','fc_rating_numeric','match_score','url','right_url' if 'right_url' in sample.columns else None]].copy()
                # clean columns
                out = out.loc[:, ~out.columns.isnull()]
                st.dataframe(out)

    st.success("Analysis complete.")

# If not run yet, show examples and instructions
if not run_scrape:
    st.info("Set options in the sidebar and click **Scrape & Compare** to start.")
    st.markdown("**Notes & tips**:")
    st.markdown("""
    - Start with small page numbers (1-3). PolitiFact and FactCheck have dozens/hundreds of pages.
    - If you get few or no matches: increase pages or lower the fuzzy threshold.
    - This app uses fuzzy text matching — not a perfect 'claim identity' test. Claims that are reworded a lot may not match.
    - If you need reproducible datasets, add a CSV upload feature (I can add that if you want).
    """)
