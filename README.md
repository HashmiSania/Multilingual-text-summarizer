# Multilingual-text-summarizer
Flask-based Multilingual News Summarizer employs mT5 (abstractive) &amp; Sumy (extractive) techniques to condense news articles in English, Hindi, Urdu &amp; Telugu. Integrated with Google Translate API, it auto-detects and translates input, delivering concise, effective summaries.

## 📰 Multilingual News Summarizer

An intelligent and user-friendly web application that generates concise summaries of news articles in **English, Hindi, Urdu, and Telugu**. Built with Python, Flask, and Streamlit, this project combines the power of **Natural Language Processing (NLP)** with **multilingual support** to deliver both **abstractive** and **extractive** summaries, along with automatic translation.

---

### 🚀 Features

* 🧠 **Abstractive Summarization** using the **mT5-small** model
* ✂️ **Extractive Summarization** using the **Sumy** library
* 🌍 **Language Support**: English, Hindi, Urdu, Telugu
* 🔁 **Auto-translation** of input and output using Google Translate API
* 🌐 **Flask API + Streamlit UI** for seamless backend and frontend integration
* 📉 Always ensures **abstractive summaries are shorter** than extractive ones

---

### 💻 Technologies Used

* **Python**
* **Flask** (API backend)
* **mT5-small** (multilingual text-to-text transformer)
* **Sumy** (extractive summarization)
* **Google Translate API**
* **scikit-learn, NLTK, Transformers, Torch**


### 📌 How It Works

1. **Input**: Paste a news article or upload content (supports multiple languages).
2. **Processing**:

   * Detects input language
   * Translates to English
   * Generates extractive and abstractive summaries
3. **Output**:

   * Translates the summary into the user’s selected output language
   * Displays both types of summaries with comparison


### 🧪 Use Cases

* Fast content digestion for multi-language news readers
* Language learners exploring comparative summaries
* NLP experimentation in multilingual environments
* Backend summarization API for news aggregation apps


### 🙋‍♀️ About Me

I'm **Sania Hashmi**, a final-year computer engineering student passionate about Python, NLP, and building inclusive digital tools. This project reflects my interest in **language technology**, **AI**, and **socially relevant applications**.
