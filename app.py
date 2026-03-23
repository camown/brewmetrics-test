import streamlit as st
import pandas as pd
import re
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="BREWMETRICS", page_icon="☕", layout="wide")

# ====================== PROFESSIONAL COFFEE THEME ======================
st.markdown("""
<style>
    .main {background-color: #2C1E16; color: #F5EDE4;}
    h1, h2, h3 {color: #D4A017; font-family: 'Georgia';}
    .metric-card {background-color: #3C2F2F; padding: 20px; border-radius: 15px; border: 2px solid #D4A017; text-align: center;}
    .big-score {font-size: 4.5rem; font-weight: bold; color: #D4A017;}
    .stButton>button {background-color: #D4A017; color: #2C1E16; font-weight: bold; border-radius: 10px;}
    .viral-green {color: #22C55E;}
    .viral-red {color: #EF4444;}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("https://i.imgur.com/8QJ5z0k.png", width=200)
    st.markdown("<h1 style='text-align:center; margin-bottom:0;'>BREWMETRICS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.3rem; color:#D4A017;'>AI Viral Score Predictor for Cavite Coffee Shops</p>", unsafe_allow_html=True)
    st.caption("Custom VADER + Random Forest • Trained on 530+ real Imus posts • Thesis 2025")

# ====================== NAVIGATION TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "🔥 BrewViral Predictor", "📊 Sentiment Analyzer", "☕ Shop Insights", "💡 Recommendations"])

# ====================== TAB 1: HOME ======================
with tab1:
    st.subheader("Prioritized Imus Coffee Shops")
    priority_data = {
        "Shop Name": ["Rojo Cafe", "Sounds Like Coffee", "TASA", "D'Kalidad"],
        "Priority": [1, 2, 3, 4],
        "Avg. Engagement": [1564.85, 482.13, 215.45, 131.33],
        "Key Focus": ["Viral reels & reach", "Niche high-quality content", "Conversation & comments", "Growth tracking"]
    }
    st.dataframe(pd.DataFrame(priority_data), use_container_width=True)

    st.markdown("**Thesis Stats**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts Analyzed", "530+")
    c2.metric("Custom VADER Accuracy", "77.27%")
    c3.metric("Model Accuracy (SMOTE)", "59%")
    c4.metric("Shops Covered", "6")

# ====================== TAB 2: BREWVIRAL PREDICTOR (YOUR MAIN TOOL) ======================
with tab2:
    st.subheader("Free BrewViral Score Predictor")
    st.caption("Paste any caption → get instant 0-100 Viral Score (probability of High engagement)")

    left, right = st.columns([1, 1])

    with left:
        media_type = st.selectbox("Media Type", ["Image", "Video/Reel"])
        is_video = 1 if "Video" in media_type else 0
        caption = st.text_area("Paste your caption here...", height=180,
                               placeholder="Valentine’s Day at Rojo Cafe 💐💖 Double the love, half the price!")
        followers = st.number_input("Shop followers", min_value=500, value=6000, step=500)

    with right:
        if st.button("🚀 Predict Viral Score", type="primary", use_container_width=True):
            if not caption.strip():
                st.error("Please paste a caption")
                st.stop()

            with st.spinner("Brewing AI analysis..."):
                # Custom VADER
                sia = SentimentIntensityAnalyzer()
                with open('cavite_lexicon.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            word, score = line.split(':')
                            sia.lexicon[word.strip()] = float(score.strip())
                sentiment_score = sia.polarity_scores(caption)['compound']

                # Features
                caption_length = len(caption)
                emoji_count = len(re.findall(r'[\U0001F000-\U0001FFFF]', caption))
                hashtag_count = caption.count('#')
                has_promo = 1 if re.search(r'promo|sale|discount|buy 2|grab now|half price|double', caption, re.I) else 0
                is_question = 1 if '?' in caption else 0

                # Model prediction
                model = joblib.load('engagement_model.pkl')
                le_post = joblib.load('le_post.pkl')
                le_media = joblib.load('le_media.pkl')

                input_df = pd.DataFrame([{
                    'caption_length': caption_length,
                    'sentiment_score': sentiment_score,
                    'is_video': is_video,
                    'has_promo': has_promo,
                    'is_question': is_question,
                    'emoji_count': emoji_count,
                    'hashtag_count': hashtag_count,
                    'comment_count': 0,
                    'follower_count_at_collection': followers,
                    'post_type_encoded': le_post.transform(['post'])[0],
                    'media_type_encoded': le_media.transform([media_type])[0]
                }])

                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                score = round(proba[2] * 100) if len(proba) > 2 else round(max(proba) * 100)

                # Display
                color = "viral-green" if score >= 70 else "viral-red" if score < 40 else ""
                st.markdown(f'<div class="metric-card"><span class="big-score {color}">{score}</span><br><b>{pred.upper()} ENGAGEMENT</b></div>', unsafe_allow_html=True)

                st.subheader("Why this score?")
                cols = st.columns(4)
                cols[0].metric("Sentiment", f"{round(sentiment_score,2)}")
                cols[1].metric("Emojis", emoji_count)
                cols[2].metric("Promo/Question", "Yes" if has_promo or is_question else "No")
                cols[3].metric("Length", caption_length)

                st.subheader("Actionable Tips (Thesis-backed)")
                tips = []
                if emoji_count < 3: tips.append("✅ Add 2–3 emojis → +42% engagement")
                if not is_question: tips.append("✅ Add a question → stronger hook")
                if sentiment_score < 0.6: tips.append("✅ Use Taglish words: sarap, ganda, sulit")
                if not has_promo: tips.append("✅ Add a limited offer")
                for t in tips: st.success(t)

# ====================== TAB 3: SENTIMENT ANALYZER ======================
with tab3:
    st.subheader("Real-time Custom VADER Sentiment Analyzer")
    text = st.text_area("Paste any caption or comment", height=150)
    if text:
        sia = SentimentIntensityAnalyzer()
        with open('cavite_lexicon.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    word, score = line.split(':')
                    sia.lexicon[word.strip()] = float(score.strip())
        score = sia.polarity_scores(text)['compound']
        cat = "Positive" if score > 0.1 else "Negative" if score < -0.05 else "Neutral"
        st.metric("Cavite-Tuned Sentiment Score", f"{score:.3f}", cat)

# ====================== TAB 4: SHOP INSIGHTS ======================
with tab4:
    st.subheader("Imus Coffee Shop Performance")
    st.info("Static insights from your thesis data (live model predictions coming in v2)")
    st.markdown("- **Rojo Cafe**: Highest viral potential (1,564 avg engagement)")
    st.markdown("- **TASA**: Conversation king (65 comments average)")
    st.markdown("- Videos + emojis = strongest predictor of High engagement")

# ====================== TAB 5: RECOMMENDATIONS ======================
with tab5:
    st.subheader("Pro Tips for Cavite Coffee Shops")
    st.markdown("• Post Reels on weekends → highest predicted score")
    st.markdown("• Use 3–5 emojis + 1 question → +35–42% engagement")
    st.markdown("• Taglish + local words (sarap, ganda, sulit) = better VADER accuracy")
    st.success("Deploy this dashboard and share the link with shop owners!")

st.divider()
st.caption("BREWMETRICS • John Paul M. Fidelson + Team • Cavite State University-Imus • October 2025")
