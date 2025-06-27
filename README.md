# 🧠 Mental Health Chatbot (Multilingual)

A simple, console-based chatbot that helps answer mental health questions using real responses from licensed therapists. It supports multiple languages by combining semantic search with automatic translation.

---

## 📌 Features

- 🔍 Semantic question-answer matching using SentenceTransformers
- 🌐 Multilingual support using a multilingual transformer model
- 🌎 Language detection and translation with Deep Translator
- 📚 Based on real counseling data from [CounselChat](https://counselchat.com)
- 💬 Easy-to-use command-line interface

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/natej-ghodbane/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Ensure the cleaned file `counselchat_final.csv` is in the same directory.
This file should contain the columns:
- `questionTitle`
- `questionText`
- `topics`
- `answerText`

You can generate this from the original `counselchat-data.csv` using a cleaning script.

### 4. Run the Chatbot

```bash
python chatbot.py
```

---

## 💬 Example Usage

```
🧠 Mental Health Chatbot (Multilingual) — type 'exit' to quit

👤 You: I feel very anxious and can't sleep.
🤖 It's normal to feel overwhelmed. Here are a few things you can try...

👤 You: exit
👋 Take care!
```

---

## 🧰 Tech Stack

- `pandas` – for loading and managing the dataset
- `sentence-transformers` – for multilingual question embeddings
- `torch` – backend for model inference
- `langdetect` – detects user input language
- `deep-translator` – translates answers into user’s native language

---

## 📁 File Structure

```
.
├── chatbot.py              # Main chatbot script
├── counselchat_final.csv   # Cleaned mental health Q&A dataset
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚠️ Disclaimer

This chatbot is not a replacement for professional mental health care.  
It is intended for educational or supportive purposes only.  
If you're in crisis or need urgent help, please seek assistance from a licensed therapist or emergency services.

---

## 🙋‍♀️ Author

Built with ❤️ by Natej Ghodbane 
Feel free to contribute, report issues, or fork the project!
