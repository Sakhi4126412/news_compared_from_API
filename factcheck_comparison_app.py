"""
app/factcheck_comparison_app.py
Streamlit app using Google Fact Check Tools API for claim data.
Replaces Snopes scraper with API access.
"""

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import logging

from utils import normalize_rating, match_claims

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# CONFIG
API_KEY = "<YOUR_GOOGLE_API_KEY_HERE>"
GOOGLE_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# ----------------------------
# Streamlit setup
st.set_page_config(page_title="Fact-Checker Comparison via Google API", layout="wide")
st.title("📰 Fact-Checker Truthfulness Comparison (via Google API)")
st.markdown("""
This app uses the **Google Fact Check Tools API** to retrieve fact-check claims, then compares truth-rating consistency.
""")

# ----------------------------
# Data fetching
with st.spinner("Fetching fact-check data from Google Fact Check API..."):
    try:
        params = {
            "key": API_KEY,
            "query": "",            # you may specify a query term, or leave blank for recent claims
            "languageCode": "en-US",
            "pageSize": 50
        }
        resp = requests.get(GOOGLE_API_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("claims", [])
        df_google = pd.DataFrame(items)
    except Exception as e:
        logger.exception("API request failed")
        st.error(f"Failed to fetch from Google Fact Check API: {e}")
        df_google = pd.DataFrame()

st.success("✅ Data fetch attempt completed")

# ----------------------------
# Process Google Data to unified format
def process_google_df(df: pd.DataFrame) -> pd.DataFrame:
    # We expect fields like: claimText, textualRating, author.name, datePublished
    processed = pd.DataFrame()
    if df.empty:
        return processed

    # Extract relevant columns
    processed["claim"] = df.get("text", df.get("claimText", None))
    processed["source"] = df.get("claimReview", [{}]).apply(
        lambda cr: (cr.get("publisher", {}).get("name") if isinstance(cr, dict) else None)
    )
    processed["rating"] = df.get("claimReview", [{}]).apply(
        lambda cr: (cr.get("textualRating") if isinstance(cr, dict) else None)
    )
    # Normalize ratings to numeric
    processed["rating_numeric"] = processed["rating"].apply(lambda r: normalize_rating(r) if isinstance(r, str) else None)
    # Drop rows without claim or rating
    processed = processed.dropna(subset=["claim", "rating_numeric"])
    return processed

df_google_processed = process_google_df(df_google)

# For comparison, you might still need a second dataset. 
# If you have another source, fetch and process it here (for example via scraping or another API).
# For demo, we’ll use Google data grouped by publisher vs publisher for comparison.
st.header("🔍 Sample of Google Fact Check Data")
if df_google_processed.empty:
    st.warning("No data available from Google API to display.")
else:
    st.dataframe(df_google_processed.head(10))

# ----------------------------
# Create comparison dataset
st.header("⚖️ Generating comparison dataset")
# Example: compare claims from two different publishers within the Google data
if df_google_processed.shape[0] < 2:
    st.warning("Not enough data to compare two different sources.")
    df_compare = pd.DataFrame()
else:
    # pick two major publishers
    publishers = df_google_processed["source"].value_counts().index[:2].tolist()
    if len(publishers) < 2:
        st.warning("Not enough distinct publishers in data for comparison.")
        df_compare = pd.DataFrame()
    else:
        pub1, pub2 = publishers[0], publishers[1]
        df1 = df_google_processed[df_google_processed["source"] == pub1].rename(
            columns={"rating": f"rating_{pub1}", "rating_numeric": f"rating_num_{pub1}"}
        ).loc[:, ["claim", f"rating_num_{pub1}"]]
        df2 = df_google_processed[df_google_processed["source"] == pub2].rename(
            columns={"rating": f"rating_{pub2}", "rating_numeric": f"rating_num_{pub2}"}
        ).loc[:, ["claim", f"rating_num_{pub2}"]]
        # Merge on claim text (inner join)
        df_compare = pd.merge(df1, df2, on="claim", how="inner")
        if df_compare.empty:
            st.info(f"No overlapping claims found between {pub1} and {pub2}.")
        else:
            st.success(f"Comparison dataset between {pub1} and {pub2} created.")
            st.dataframe(df_compare.head(10))

# ----------------------------
# Correlation & plot
if not df_compare.empty:
    s_col = f"rating_num_{pub1}"
    p_col = f"rating_num_{pub2}"
    numeric_df = df_compare[[s_col, p_col]].dropna()
    if numeric_df.shape[0] >= 2:
        corr, _ = pearsonr(numeric_df[s_col], numeric_df[p_col])
        st.metric("Correlation", f"{corr:.2f}")
        fig, ax = plt.subplots()
        ax.scatter(numeric_df[s_col], numeric_df[p_col])
        ax.set_xlabel(f"{pub1} rating (numeric)")
        ax.set_ylabel(f"{pub2} rating (numeric)")
        ax.set_title("Truthfulness Rating Comparison")
        st.pyplot(fig)
        # Download
        csv_bytes = df_compare.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download comparison CSV",
            data=csv_bytes,
            file_name="factcheck_comparison_google.csv",
            mime="text/csv"
        )
    else:
        st.warning("Insufficient matched numeric ratings for correlation.")
else:
    st.info("No comparison dataset to show correlation.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Google Fact Check Tools API")

