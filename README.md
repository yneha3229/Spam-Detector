# 📩 Spam Detector App

A simple and interactive web application that detects whether a message is **Spam** or **Not Spam** using a trained **Random Forest classifier**. Built with [Streamlit](https://streamlit.io/) for an instant, browser-based experience.

---

## 🚀 Features

- 🔍 Predicts whether a message is spam or not
- 🧠 Uses a pre-trained Random Forest model
- 💬 Simple and clean UI built with Streamlit
- ⚡ Fast and lightweight — runs in the browser with no setup needed

---

## 🧠 Model

The Random Forest model was trained on a labeled dataset of SMS messages (ham/spam). It uses basic preprocessing (like removing stopwords, tokenization, and vectorization) to learn the difference between legitimate and spam messages.

The model is saved using `joblib` or `pickle` as `model.pkl`.

---

## 🛠️ Installation

You can run the app locally using Python and Streamlit.

### 1. Clone the repository

```bash
git clone https://github.com/yneha3229/Spam-Detector.git
cd Spam-Detector
🌐 Live App
👉 Deployed Version (if hosted on Streamlit Cloud):
https://yneha3229-spam-detector.streamlit.app
👩 Neha Yadav
BTech Engineering Physics, IIT Bombay
