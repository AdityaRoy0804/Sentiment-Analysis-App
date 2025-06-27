# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .title { font-size: 36px; font-weight: bold; color: #2e8b57; }
    .footer { position: fixed; bottom: 10px; width: 100%; text-align: center; color: gray; font-size: 13px; }
    .stTextArea label { font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>💬 Sentiment Analysis App</div>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["💡 Real-Time Prediction", "📊 Batch Prediction"])

# -------------------------------------------
# ✅ TAB 1 — Real-Time Sentiment Prediction
# -------------------------------------------
with tab1:
    st.subheader("🔍 Real-Time Sentiment Analyzer")
    user_input = st.text_area("📝 Enter your product review:")

    if st.button("📊 Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            pred = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)[0]

            # Show prediction
            if pred == "positive":
                st.success(f"✅ Sentiment: **Positive** 😊")
            elif pred == "negative":
                st.error(f"❌ Sentiment: **Negative** 😠")
            else:
                st.info(f"😐 Sentiment: **Neutral** 😐")

            st.markdown("**Prediction Confidence (%):**")
            st.write({
                model.classes_[i]: round(prob[i]*100, 2)
                for i in range(len(model.classes_))
            })

# -------------------------------------------
# ✅ TAB 2 — Batch File Upload
# -------------------------------------------
with tab2:
    st.subheader("📁 Upload File for Batch Sentiment Analysis")
    st.markdown("📝 **Note:** Please upload a `.csv` or `.xlsx` file with:")
    st.markdown("""
    - **Column 1**: `Review` or `Summary` (text to analyze)  
    - **Column 2**: `Sentiment` (optional, ignored in analysis)  
    """)

    file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if file:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            if df.shape[1] < 1:
                st.error("❌ File must contain at least one column with text.")
            else:
                # Pick first text column
                text_col = df.columns[0]
                df['cleaned'] = df[text_col].astype(str).apply(clean_text)
                vecs = vectorizer.transform(df['cleaned'])
                df['Predicted_Sentiment'] = model.predict(vecs)

                st.success("✅ Sentiment analysis complete.")
                
                # Display result summary
                sentiment_counts = df['Predicted_Sentiment'].value_counts(normalize=True) * 100
                st.markdown("### 📊 Sentiment Distribution:")
                st.write(sentiment_counts.round(2).to_frame('%'))

                # Option to download results
                st.markdown("📥 Download the file with predictions:")
                result_csv = df[[text_col, 'Predicted_Sentiment']].to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download CSV", result_csv, "sentiment_results.csv", "text/csv")

        
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# Footer
st.markdown("<div class='footer'>Made with ❤️ by Aditya | Streamlit + NLP + ML</div>", unsafe_allow_html=True)
