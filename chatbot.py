import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Load CSV
df = pd.read_csv("counselchat_final.csv")  
questions = df["questionText"].tolist()
answers = df["answerText"].tolist()

# Load multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Let the user choose their preferred language
print("🌍 Choose your preferred language (e.g., en, fr, ar, es, de...):")
preferred_lang = input("🌐 Language code [default is 'en']: ").strip().lower()
if not preferred_lang:
    preferred_lang = 'en'

print(f"\n🧠 Mental Health Chatbot — Type 'exit' to quit. Responding in: {preferred_lang.upper()}\n")

# Chat loop
while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "exit":
        print("👋 Take care!")
        break

    # Get embedding and similarity score
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score > 0.5:
        answer_en = answers[best_idx]

        # Translate answer to preferred language if not English
        if preferred_lang != 'en':
            try:
                translated = GoogleTranslator(source='en', target=preferred_lang).translate(answer_en)
                print(f"🤖 {translated} (translated from English)\n")
            except Exception:
                print(f"🤖 (EN) {answer_en} (⚠ translation failed)\n")
        else:
            print(f"🤖 {answer_en}\n")
    else:
        print("🤖 I'm sorry, I couldn't find a suitable answer.\n")
