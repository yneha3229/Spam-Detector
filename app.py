import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt

# ----- Page Settings -----
st.set_page_config(
    page_title="📩 SMS Spam Detector",
    page_icon="📨",
    layout="centered"
)

# ----- Load Model & Vectorizer -----
model = joblib.load("rf_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer_spam.pkl")

# ----- Clean Function -----
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r'\[.*?\]', '', text)  # Remove [1]
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ----- Sidebar -----
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5610/5610944.png", width=100)
    st.markdown("### 👩‍💻 Made by Neha Yadav")
    st.markdown("Built with `Python`, `Streamlit`, and `Random Forest Classifier`")
    st.markdown("### 📊 Dataset")
    st.write("UCI SMS Spam Collection Dataset")
    st.markdown("---")
    st.markdown("🔗 [View on GitHub](https://github.com/yneha3229)")

# ----- Main App -----
st.image("spam.png", use_container_width=True)

st.title("📩 SMS Spam Detector")
st.markdown("_Enter an SMS message to check if it's spam or not._")

# ----- Input Area -----
input_text = st.text_area("✍️ Paste your SMS message here:")

# ----- Predict -----
if st.button("🔍 Detect SMS"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][pred]

        # ----- Confidence Bar -----
        st.markdown("### 🔢 Confidence Level")
        fig, ax = plt.subplots(figsize=(4, 0.5))
        ax.barh(["Confidence"], [proba], color=("green" if pred == 0 else "red"))
        ax.set_xlim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)

        # ----- Prediction Output -----
        if proba >= 0.85:

            st.success(f"✅ This message is **NOT SPAM** with **{proba*100:.2f}%** confidence.")
        else:
            st.error(f"❌ This message is **SPAM** with **{proba*100:.2f}%** confidence.")

        # ----- Download Result -----
        report = f"Prediction: {'NOT SPAM' if pred == 0 else 'SPAM'}\nConfidence: {proba*100:.2f}%\n\nMessage:\n{input_text}"
        st.download_button("📥 Download Result", data=report, file_name="sms_result.txt")

# ----- Footer -----
st.markdown("---")
st.markdown("⚠️ _This is a demo app for educational purposes only. Don't use it for sensitive decisions._")
