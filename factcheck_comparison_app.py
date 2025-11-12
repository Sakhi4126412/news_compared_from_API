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

# Google Fact Check Highlights
st.subheader("📚 Google Fact Check Highlights")
google_highlights = [
    {
        "claim": "Brazilian hairdresser Larissa Nery called Rahul Gandhi a ‘failed politician’.",
        "rating": "False",
        "source": "India Today",
        "url": "https://toolbox.google.com/factcheck/explorer"
    },
    {
        "claim": "Dehradun University awarded 50 marks for attending PM Modi’s rally.",
        "rating": "False",
        "source": "BOOM",
        "url": "https://toolbox.google.com/factcheck/explorer"
    },
    {
        "claim": "Drunk man pets Bengal Tiger in Pench, Madhya Pradesh.",
        "rating": "False (AI-generated)",
        "source": "DigitEye India",
        "url": "https://toolbox.google.com/factcheck/explorer"
    }
]

for item in google_highlights:
    st.markdown(f"**Claim:** {item['claim']}")
    st.markdown(f"- **Rating:** {item['rating']}")
    st.markdown(f"- **Source:** {item['source']}")
    st.markdown(f"- [View Source]({item['url']})")

# ClaimBuster Highlights (Static examples)
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
    google_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={GOOGLE_API_KEY}"
    g_response = requests.get(google_url)

    if g_response.status_code == 200:
        claims = g_response.json().get("claims", [])
        if claims:
            for item in claims:
                st.markdown(f"**Claim:** {item.get('text')}")
                for review in item.get("claimReview", []):
                    st.markdown(f"- **Rating:** {review.get('textualRating')}")
                    st.markdown(f"- **Publisher:** {review['publisher'].get('name')}")
                    st.markdown(f"- [Read more]({review.get('url')})")
        else:
            st.warning("No fact-checks found for this claim.")
    else:
        st.error("Google Fact Check API error.")
