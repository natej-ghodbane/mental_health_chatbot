import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect


df = pd.read_csv("counselchat_final.csv")  
questions = df["questionText"].tolist()
answers = df["answerText"].tolist()

# === Multilingual Embeddings ===
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

print("\n🧠 Mental Health Chatbot (Multilingual) — type 'exit' to quit\n")

while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "exit":
        print("👋 Take care!")
        break

    # Detect language
    user_lang = detect(user_input)

    # Embed and compare
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score > 0.5:
        answer_en = answers[best_idx]
        # Translate (if user input not in English)
        if user_lang != 'en':
            try:
                translated = GoogleTranslator(source='en', target=user_lang).translate(answer_en)
                print(f"🤖 {translated} (translated from English)\n")
            except Exception as e:
                print(f"🤖 (EN) {answer_en} (⚠ translation failed)\n")
        else:
            print(f"🤖 {answer_en}\n")
    else:
        print("🤖 I'm sorry, I couldn't find a suitable answer.\n")
