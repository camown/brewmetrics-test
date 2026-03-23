import streamlit as st
import pandas as pd
import re
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="BREWMETRICS - Viral Predictor", page_icon="☕", layout="wide")

# Ultra-premium coffee dark theme (inspired by modern SaaS + your ref designs)
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a0f0a 0%, #2c1e16 100%);
        color: #f5ede4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: transparent;
        border-bottom: 2px solid #3c2f2f;
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        white-space: nowrap;
        background: #3c2f2f;
        border-radius: 12px 12px 0 0;
        color: #d4a017;
        font-weight: 600;
        padding: 0 32px;
        margin-right: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #d4a017 !important;
        color: #1a0f0a !important;
    }
    h1 {color: #d4a017; text-align: center; font-size: 3.2rem; margin: 0.5rem 0;}
    h2, h3 {color: #d4a017;}
    .big-score {
        font-size: 6rem !important;
        font-weight: 900;
        line-height: 1;
        text-shadow: 0 4px 12px rgba(212,160,23,0.4);
    }
    .card {
        background: rgba(60,47,47,0.65);
        backdrop-filter: blur(12px);
        border: 1px solid #d4a01733;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .step {
        background: #3c2f2f;
        border-left: 5px solid #d4a017;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
    }
    .platform-btn {
        background: #3c2f2f;
        color: #f5ede4;
        border: 2px solid #d4a01744;
        border-radius: 12px;
        padding: 12px 24px;
        margin: 8px;
        font-weight: bold;
    }
    .platform-btn:hover {border-color: #d4a017;}
    .platform-btn.selected {background: #d4a017; color: #1a0f0a;}
</style>
""", unsafe_allow_html=True)

# Header - Thesis branding clean & centered
st.markdown("<h1>BREWMETRICS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.4rem; color:#d4a017; margin-bottom:0;'>VADER + Predictive Viral Score for Cavite Coffee Shops</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1rem; color:#a67c00;'>Kyle Andrei Escauriaga • John Paul M. Fidelson • Henry Luis P. Pula • October 2025</p>", unsafe_allow_html=True)

tabs = st.tabs(["Home", "Predict Virality", "Sentiment Analyzer", "Shop Insights", "Tips"])

# Home - Priority table + stats (card style)
with tabs[0]:
    st.subheader("Prioritized Imus Shops for Data Collection")
    
    priority_data = [
        {"Shop": "Rojo Cafe", "Prio": 1, "Posts": 20, "Avg Eng": "1,564.85", "Comments": 53, "Focus": "Highest engagement – viral reels & reach"},
        {"Shop": "Sounds Like Coffee", "Prio": 2, "Posts": 15, "Avg Eng": "482.13", "Comments": 43, "Focus": "High impact from fewer posts – niche quality"},
        {"Shop": "TASA", "Prio": 3, "Posts": 20, "Avg Eng": "215.45", "Comments": 65, "Focus": "Conversation hub – most comments, community testing"},
        {"Shop": "D'Kalidad", "Prio": 4, "Posts": 15, "Avg Eng": "131.33", "Comments": 44, "Focus": "Growth tracker – reach vs active engagement"}
    ]
    
    for shop in priority_data:
        with st.container():
            st.markdown(f"""
            <div class="card">
                <h3>{shop['Shop']} (Priority {shop['Prio']})</h3>
                <p><strong>Avg Engagement:</strong> {shop['Avg Eng']} • <strong>Comments:</strong> {shop['Comments']}</p>
                <p><strong>Focus:</strong> {shop['Focus']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("Thesis Quick Stats")
    cols = st.columns(4)
    cols[0].metric("Posts", "530+")
    cols[1].metric("Custom VADER Acc.", "77.3%")
    cols[2].metric("RF Acc. (SMOTE)", "59%")
    cols[3].metric("Shops", "6")

# Predict Virality - Closer to your ref (big score, steps, tips)
with tabs[1]:
    st.subheader("AI Viral Score Predictor")
    st.markdown("Paste caption → instant 0-100 viral probability (trained on real Imus coffee posts)")

    platform = st.radio("Platform Context", ["Instagram", "Facebook", "TikTok", "Other"], horizontal=True, key="plat")
    
    caption = st.text_area("Your Post Caption (emojis + hashtags OK)", height=140,
                           placeholder="Valentine’s Day at Rojo Cafe 💐💖 B1G1 on all lattes! Who's coming? ☕❤️")
    
    followers = st.slider("Shop Followers (approx)", 1000, 20000, 6000, 500)

    if st.button("🚀 Predict Viral Score", type="primary", use_container_width=True):
        if caption:
            with st.spinner("Brewing prediction..."):
                sia = SentimentIntensityAnalyzer()
                try:
                    with open('cavite_lexicon.txt', 'r', encoding='utf-8') as f:
                        for line in f:
                            if ':' in line and not line.startswith('#'):
                                w, s = line.split(':', 1)
                                sia.lexicon[w.strip()] = float(s.strip())
                except:
                    pass

                vscore = sia.polarity_scores(caption)['compound']
                length = len(caption)
                emojis = len(re.findall(r'[\U0001F000-\U0001FFFF]', caption))
                hashtags = caption.count('#')
                promo = 1 if any(k in caption.lower() for k in ['promo','sale','b1g1','grab','special']) else 0
                question = 1 if '?' in caption else 0
                is_video = 1 if platform in ["Instagram", "TikTok"] else 0

                input_df = pd.DataFrame([{
                    'caption_length': length, 'sentiment_score': vscore, 'is_video': is_video,
                    'has_promo': promo, 'is_question': question, 'emoji_count': emojis,
                    'hashtag_count': hashtags, 'comment_count': 0,
                    'follower_count_at_collection': followers,
                    'post_type_encoded': 0, 'media_type_encoded': is_video
                }])

                model = joblib.load('engagement_model.pkl')
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                viral_score = round(proba[list(model.classes_).index('High')] * 100 if 'High' in model.classes_ else max(proba) * 100)

                color = "#22c55e" if viral_score >= 70 else "#f59e0b" if viral_score >= 40 else "#ef4444"
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <div class="big-score" style="color:{color}">{viral_score}</div>
                    <h2>{pred.upper()} VIRAL POTENTIAL</h2>
                    <p style="font-size:1.2rem;">Platform: {platform}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Why this score?")
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", f"{vscore:.2f}")
                c2.metric("Hooks", f"{emojis} emojis • {'Yes' if question else 'No'} question")
                c3.metric("Length", f"{length} chars")

                st.subheader("Boost Tips")
                tips = []
                if length > 120: tips.append("Shorten to <120 chars – mobile scroll is fast")
                if emojis < 3: tips.append("Add 3+ emojis – boosts dwell time & engagement")
                if not question: tips.append("End with question – sparks comments")
                if promo == 0: tips.append("Include promo/offer – higher predicted High")
                for tip in tips:
                    st.info(tip)

# Keep other tabs minimal for now – expand if needed
with tabs[2]:
    st.subheader("Custom VADER Sentiment Check")
    text = st.text_area("Test any text", height=100)
    if text and st.button("Analyze"):
        sia = SentimentIntensityAnalyzer()
        # reload lexicon if needed...
        score = sia.polarity_scores(text)['compound']
        st.metric("Score", f"{score:.3f}", "Positive" if score > 0.1 else "Neutral/Negative")

st.divider()
st.caption("BREWMETRICS • Cavite State University-Imus • Thesis October 2025 • Demo Only")
