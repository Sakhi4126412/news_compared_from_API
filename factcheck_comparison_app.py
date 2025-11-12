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
        st.error("⚠️ Please enter y
