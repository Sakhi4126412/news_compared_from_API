import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scraper import get_snopes_articles, get_politifact_articles
from utils import normalize_rating, match_claims

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Fact Checker Comparison", layout="wide")

st.title("📰 Fact-Checker Truthfulness Comparison App")
st.markdown("""
Compare the truthfulness ratings of **news claims** from **Snopes** and **PolitiFact**.
This app scrapes live fact-checks, normalizes their ratings, and visualizes consistency.
""")

# ----------------------------
# Fetch Data
# ----------------------------
with st.spinner("Fetching latest fact checks..."):
    df_snopes = get_snopes_articles(limit=20)
    df_politifact = get_politifact_articles(limit=20)

st.success("✅ Data fetched successfully!")

# Normalize ratings
df_snopes["rating_numeric"] = df_snopes["rating"].apply(normalize_rating)
df_politifact["rating_numeric"] = df_politifact["rating"].apply(normalize_rating)

# ----------------------------
# Display Data
# ----------------------------
st.header("🔎 Latest Fact-Checks")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Snopes")
    st.dataframe(df_snopes.head(10))
with col2:
    st.subheader("PolitiFact")
    st.dataframe(df_politifact.head(10))

# ----------------------------
# Compare Claims
# ----------------------------
st.header("⚖️ Matched Claims Comparison")
df_compare = match_claims(df_snopes, df_politifact)

if df_compare.empty:
    st.warning("No similar claims found between the two sources.")
else:
    st.dataframe(df_compare.head(10))

    # Correlation
    st.header("📊 Truthfulness Correlation")
    if df_compare["rating_snopes_num"].notnull().any() and df_compare["rating_politifact_num"].notnull().any():
        corr, _ = pearsonr(
            df_compare["rating_snopes_num"].dropna(),
            df_compare["rating_politifact_num"].dropna()
        )
        st.metric("Correlation between Snopes & PolitiFact", f"{corr:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(df_compare["rating_snopes_num"], df_compare["rating_politifact_num"])
        ax.set_xlabel("Snopes Rating")
        ax.set_ylabel("PolitiFact Rating")
        ax.set_title("Truthfulness Comparison")
        st.pyplot(fig)

    st.download_button(
        label="📥 Download Comparison Data (CSV)",
        data=df_compare.to_csv(index=False),
        file_name="factcheck_comparison.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Python")
