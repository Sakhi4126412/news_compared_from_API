"""
Streamlit app for comparing fact-check truthfulness ratings using Google Fact Check Tools API.
Fixed version with API key input, query support, and improved error handling.
"""

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from utils import normalize_rating
import logging

# ----------------------------
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
GOOGLE_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# ----------------------------
# Streamlit setup
st.set_page_config(page_title="Fact Checker Comparison via Google API", layout="wide")
st.title("📰 Fact-Checker Truthfulness Comparison App")
st.markdown("""
This app fetches **fact-checked claims** from the Google Fact Check Tools API  
and compares truth ratings between different publishers.
""")

# ----------------------------
# User inputs
st.sidebar.header("🔑 API Settings")
api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

st.sidebar.header("🗞️ Search Settings")
query = st.sidebar.text_input("Enter a topic or keyword", "Elections")
page_size = st.sidebar.slider("Number of claims to fetch", 10, 100, 30)

# ----------------------------
# Fetch data
if st.sidebar.button("Fetch Fact-Check Data"):
    if not api_key:
        st.error("⚠️ Please enter your Google API key in the sidebar before fetching data.")
    else:
        with st.spinner(f"Fetching claims for topic: '{query}'..."):
            try:
                params = {
                    "key": api_key,
                    "query": query,
                    "languageCode": "en-US",
                    "pageSize": page_size
                }
                resp = requests.get(GOOGLE_API_URL, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                claims = data.get("claims", [])
                if not claims:
                    st.warning("No claims found for this query. Try a different keyword.")
                else:
                    st.success(f"✅ Retrieved {len(claims)} claims successfully.")
                    df_google = pd.DataFrame(claims)

                    # ----------------------------
                    # Extract claim details
                    def extract_review_field(row, key):
                        reviews = row.get("claimReview", [])
                        if isinstance(reviews, list) and len(reviews) > 0:
                            val = reviews[0].get(key)
                            if isinstance(val, dict):
                                return val.get("name")
                            return val
                        return None

                    df_google["claim_text"] = df_google["text"]
                    df_google["source"] = df_google.apply(lambda r: extract_review_field(r, "publisher"), axis=1)
                    df_google["rating"] = df_google.apply(lambda r: extract_review_field(r, "textualRating"), axis=1)
                    df_google["url"] = df_google.apply(lambda r: extract_review_field(r, "url"), axis=1)
                    df_google["rating_numeric"] = df_google["rating"].apply(normalize_rating)

                    # Clean up data
                    df_google = df_google.dropna(subset=["claim_text", "rating_numeric"])
                    st.dataframe(df_google[["claim_text", "source", "rating", "rating_numeric"]].head(10))

                    # ----------------------------
                    # Compare between publishers
                    publishers = df_google["source"].value_counts().index[:2].tolist()
                    if len(publishers) < 2:
                        st.warning("Not enough distinct publishers to compare ratings.")
                    else:
                        pub1, pub2 = publishers[:2]
                        st.markdown(f"### Comparing publishers: `{pub1}` vs `{pub2}`")

                        df1 = df_google[df_google["source"] == pub1][["claim_text", "rating_numeric"]]
                        df2 = df_google[df_google["source"] == pub2][["claim_text", "rating_numeric"]]
                        df_merge = pd.merge(df1, df2, on="claim_text", how="inner", suffixes=(f"_{pub1}", f"_{pub2}"))

                        if df_merge.empty:
                            st.info("No overlapping claims between the two publishers.")
                        else:
                            st.success(f"✅ Found {len(df_merge)} overlapping claims for comparison.")
                            st.dataframe(df_merge.head(10))

                            # Correlation
                            numeric_df = df_merge.dropna()
                            if len(numeric_df) >= 2:
                                corr, _ = pearsonr(
                                    numeric_df[f"rating_numeric_{pub1}"],
                                    numeric_df[f"rating_numeric_{pub2}"]
                                )
                                st.metric("📊 Truthfulness Correlation", f"{corr:.2f}")

                                # Plot scatter
                                fig, ax = plt.subplots()
                                ax.scatter(
                                    numeric_df[f"rating_numeric_{pub1}"],
                                    numeric_df[f"rating_numeric_{pub2}"]
                                )
                                ax.set_xlabel(pub1)
                                ax.set_ylabel(pub2)
                                ax.set_title("Comparison of Truthfulness Ratings")
                                st.pyplot(fig)

                                # Download option
                                csv = df_merge.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "📥 Download Comparison CSV",
                                    csv,
                                    "factcheck_comparison.csv",
                                    "text/csv"
                                )
                            else:
                                st.warning("Insufficient numeric data to compute correlation.")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error: {e}")
            except Exception as e:
                st.error(f"Failed to fetch from Google Fact Check API: {e}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Google Fact Check Tools API")
