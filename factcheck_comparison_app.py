import streamlit as st
import requests

# Set your API keys here
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
CLAIMBUSTER_API_KEY = "YOUR_CLAIMBUSTER_API_KEY"

st.set_page_config(page_title="Fact Check Comparator", page_icon="🕵️", layout="centered")

st.title("🕵️ Fact Check Comparator")
st.write("Compare the truthfulness of news claims using Google Fact Check and ClaimBuster APIs.")

# -------------------------------
# 🔥 Display Top Fact-Checked News
# -------------------------------
st.header("📰 Top Fact-Checked News")

# Google Fact Check API: Get top claims
google_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?key={GOOGLE_API_KEY}"
g_response = requests.get(google_url)

if g_response.status_code == 200:
    claims = g_response.json().get("claims", [])
    if claims:
        st.subheader("📚 Google Fact Check Highlights")
        for item in claims[:5]:  # Show top 5
            st.markdown(f"**Claim:** {item.get('text')}")
            for review in item.get("claimReview", []):
                st.markdown(f"- **Rating:** {review.get('textualRating')}")
                st.markdown(f"- **Publisher:** {review['publisher'].get('name')}")
                st.markdown(f"- [Read more]({review.get('url')})")
    else:
        st.warning("No trending fact-checks found.")
else:
    st.error("Google Fact Check API error.")

# ClaimBuster does not provide trending claims, so we simulate examples
st.subheader("🔍 ClaimBuster Highlights")
claimbuster_examples = [
    "The U.S. economy added 300,000 jobs last month.",
    "COVID vaccines contain microchips.",
    "India is the fastest-growing major economy in the world."
]

for example in claimbuster_examples:
    st.markdown(f"- **Claim:** {example}")

# -------------------------------
# 🧪 Claim Verification Section
# -------------------------------
st.header("🔎 Verify a News Claim")

claim = st.text_input("Enter a news claim to verify:")

if claim:
    st.subheader("🔍 ClaimBuster Analysis")
    cb_response = requests.post(
        "https://idir.uta.edu/claimbuster/api/v1/score/text/",
        json={"input": claim},
        headers={"x-api-key": CLAIMBUSTER_API_KEY}
    )

    if cb_response.status_code == 200:
        score = cb_response.json().get("score")
        st.write(f"**Check-worthiness Score:** {score:.2f}")
        st.progress(score)
    else:
        st.error("ClaimBuster API error.")

    st.subheader("📚 Google Fact Check Results")
    google_claim_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={GOOGLE_API_KEY}"
    g_claim_response = requests.get(google_claim_url)

    if g_claim_response.status_code == 200:
        claim_results = g_claim_response.json().get("claims", [])
        if claim_results:
            for item in claim_results:
                st.markdown(f"**Claim:** {item.get('text')}")
                for review in item.get("claimReview", []):
                    st.markdown(f"- **Rating:** {review.get('textualRating')}")
                    st.markdown(f"- **Publisher:** {review['publisher'].get('name')}")
                    st.markdown(f"- [Read more]({review.get('url')})")
        else:
            st.warning("No fact-checks found for this claim.")
    else:
        st.error("Google Fact Check API error.")
