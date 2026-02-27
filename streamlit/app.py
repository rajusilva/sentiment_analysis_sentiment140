import streamlit as st
import joblib
import re
import pandas as pd

tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("log_reg_model.pkl")

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"@\w+|#\w+", "", text)       # mentions & hashtags
    text = re.sub(r"[^a-z\s]", "", text)        # punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()    # extra whitespace
    return text

# Sentiment bucket
def sentiment_bucket(score):
    if score <= 0.4:
        return "Negative"
    elif score <= 0.7:
        return "Neutral"
    else:
        return "Positive"

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("Sentiment Analyzer")
st.write("Enter a tweet to analyze the sentiment.")
user_text = st.text_area("", height=150)
if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = clean_text(user_text)
        text_tfidf = tfidf_vectorizer.transform([clean_text])

        prob = model.predict_proba(text_tfidf)[0][1]
        label = "Positive" if prob >= 0.5 else "Negative"
        bucket = sentiment_bucket(prob)

        st.success("Sentiment analysis completed!")
        st.subheader("Results")
        st.write(f"**Predicted Sentiment:** {label}")
        st.write(f"**Sentiment Score:** `{prob:.3f}`")
        st.write(f"**Sentiment bucket:** {bucket}")

# CSV upload (Bulk prediction)

st.subheader("Bulk Sentiment Analysis")

uploaded_file = st.file_uploader("Upload CSV file (must have 'text' column)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        results={
            "Predicted Sentiment":[],
            "Sentiment Score":[],
            "Sentiment Bucket":[]
        }
        for text in df['text'].fillna(""):
            cleaned_text = clean_text(text)
            text_tfidf = tfidf_vectorizer.transform([cleaned_text])

            prob = model.predict_proba(text_tfidf)[0][1]
            label = "Positive" if prob >= 0.5 else "Negative"
            bucket = sentiment_bucket(prob)

            results["Predicted Sentiment"].append(label)
            results["Sentiment Score"].append(prob)
            results["Sentiment Bucket"].append(bucket)

        for col, values in results.items():
            df[col] = values

        st.success("Bulk sentiment analysis completed!")

        st.dataframe(df.head())
        
        st.download_button(
        "Download Results",
         df.to_csv(index=False),
        "sentiment_analysis.csv",
        "text/csv"
            )