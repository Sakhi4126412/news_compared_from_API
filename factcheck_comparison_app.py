"""
app/factcheck_comparison_app.py
Robust Streamlit app entrypoint for Fact-Checker Truthfulness Comparison.
This version defends against missing columns (KeyError) from scraped data,
and gives helpful warnings instead of crashing.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import logging

# local modules (keep as before)
from scraper import get_snopes_articles, get_politifact_articles
from utils import normalize_rating, match_claims

# configure simple logging to console (helpful for debugging on Streamlit Cloud)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Fact Checker Comparison", layout="wide")
st.title("📰 Fact-Checker Truthfulness Comparison App (robust)")
st.markdown(
    """
    Compare the truthfulness ratings of news claims from **Snopes** and **PolitiFact**.
    This version includes defensive checks for missing columns and improved error handling.
    """
)

# ----------------------------
# Fetch Data (with defensive checks)
# ----------------------------
with st.spinner("Fetching latest fact checks..."):
    try:
        df_snopes = get_snopes_articles(limit=20)
    except Exception as e:
        logger.exception("Failed to fetch Snopes articles")
        st.error(f"Failed to fetch Snopes data: {e}")
        df_snopes = pd.DataFrame()

    try:
        df_politifact = get_politifact_articles(limit=20)
    except Exception as e:
        logger.exception("Failed to fetch PolitiFact articles")
        st.error(f"Failed to fetch PolitiFact data: {e}")
        df_politifact = pd.DataFrame()

st.success("✅ Fetch attempt finished")

# ----------------------------
# Defensive normalization function wrapper
# ----------------------------
def safe_normalize_rating(raw):
    """
    Normalize rating safely:
     - Handle missing columns / None
     - Convert non-string to string where possible
     - Return None when mapping not possible
    """
    try:
        if raw is None:
            return None
        # If a pandas float (NaN) or other, convert to string only if it's not NaN
        if isinstance(raw, float) and pd.isna(raw):
            return None
        raw_str = str(raw).strip()
        if raw_str == "":
            return None
        return normalize_rating(raw_str)
    except Exception as e:
        logger.exception("normalize_rating failed for input: %r", raw)
        return None

# ----------------------------
# Ensure columns exist and create 'rating' fallback if necessary
# ----------------------------
def ensure_rating_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Make sure a DataFrame has a 'rating' column.
    If not present, attempt to find likely candidates, or create a default 'Unknown'.
    """
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    # if rating exists already, return
    if "rating" in df.columns:
        return df

    # Attempt common alternate column names
    alt_names = ["fact_rating", "rating_text", "verdict", "truth_rating", "label"]
    for alt in alt_names:
        if alt in df.columns:
            df = df.rename(columns={alt: "rating"})
            logger.info("Renamed column %s -> rating for %s", alt, source_name)
            return df

    # If there is an article URL or metadata that might contain rating, we could attempt to extract,
    # but for now create the column filled with None so downstream code does not KeyError.
    logger.warning("No 'rating' column found for %s; filling with None", source_name)
    df["rating"] = None
    return df

df_snopes = ensure_rating_column(df_snopes, "Snopes")
df_politifact = ensure_rating_column(df_politifact, "PolitiFact")

# ----------------------------
# Compute numeric ratings safely
# ----------------------------
# If rating_numeric already exists, leave it; otherwise create it using safe_normalize_rating
if "rating_numeric" not in df_snopes.columns:
    df_snopes["rating_numeric"] = df_snopes["rating"].apply(safe_normalize_rating)
else:
    # ensure values are either numeric or None
    df_snopes["rating_numeric"] = df_snopes["rating_numeric"].apply(
        lambda x: x if pd.api.types.is_number(x) else safe_normalize_rating(x)
    )

if "rating_numeric" not in df_politifact.columns:
    df_politifact["rating_numeric"] = df_politifact["rating"].apply(safe_normalize_rating)
else:
    df_politifact["rating_numeric"] = df_politifact["rating_numeric"].apply(
        lambda x: x if pd.api.types.is_number(x) else safe_normalize_rating(x)
    )

# ----------------------------
# Display Data (safe)
# ----------------------------
st.header("🔎 Latest Fact-Checks (samples)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Snopes (sample)")
    if df_snopes.empty:
        st.warning("No Snopes data available.")
    else:
        st.dataframe(df_snopes.head(10))

with col2:
    st.subheader("PolitiFact (sample)")
    if df_politifact.empty:
        st.warning("No PolitiFact data available.")
    else:
        st.dataframe(df_politifact.head(10))

# ----------------------------
# Compare Claims (defensive)
# ----------------------------
st.header("⚖️ Matched Claims Comparison (fuzzy)")

try:
    df_compare = match_claims(df_snopes, df_politifact)
except Exception as e:
    logger.exception("match_claims failed")
    st.error(f"Error when matching claims: {e}")
    df_compare = pd.DataFrame()

if df_compare.empty:
    st.info("No matched claims to display. This can happen if there are no similar claims between the sources, "
            "or if the scrapers returned empty results.")
else:
    st.dataframe(df_compare.head(20))

    # ----------------------------
    # Correlation & Visualization (safe)
    # ----------------------------
    st.header("📊 Truthfulness Correlation")
    # pick numeric columns if they exist
    s_col = "rating_snopes_num" if "rating_snopes_num" in df_compare.columns else "rating_snopes_num"
    p_col = "rating_politifact_num" if "rating_politifact_num" in df_compare.columns else "rating_politifact_num"

    # ensure columns exist and have numeric values
    if s_col in df_compare.columns and p_col in df_compare.columns:
        # drop rows where either is null
        numeric_df = df_compare[[s_col, p_col]].dropna()
        if numeric_df.shape[0] >= 2:
            try:
                corr, _ = pearsonr(numeric_df[s_col], numeric_df[p_col])
                st.metric("Correlation between Snopes & PolitiFact", f"{corr:.2f}")
                fig, ax = plt.subplots()
                ax.scatter(numeric_df[s_col], numeric_df[p_col])
                ax.set_xlabel("Snopes Rating (numeric)")
                ax.set_ylabel("PolitiFact Rating (numeric)")
                ax.set_title("Truthfulness Comparison")
                st.pyplot(fig)
            except Exception as e:
                logger.exception("Failed to compute correlation/plot")
                st.warning("Could not compute correlation or plot: %s", e)
        else:
            st.warning("Not enough matched numeric ratings to compute correlation (need at least 2).")
    else:
        st.warning("Comparison dataframe does not contain numeric rating columns.")

    # Download button (safe: only if df_compare exists)
    try:
        csv_bytes = df_compare.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Comparison Data (CSV)",
            data=csv_bytes,
            file_name="factcheck_comparison.csv",
            mime="text/csv"
        )
    except Exception as e:
        logger.exception("Failed to create download button")
        st.warning("Download not available: %s", e)

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Python — improved error handling added.")
