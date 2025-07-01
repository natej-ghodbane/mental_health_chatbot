import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("counselchat_final.csv")
    return df["questionText"].tolist(), df["answerText"].tolist()

# Load embeddings
@st.cache_resource
def load_model_and_embeddings(questions):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(questions, convert_to_tensor=True)
    return model, embeddings

# Main app
def main():
    st.set_page_config(page_title="🧠 Mental Health Chatbot", layout="centered")
    st.title("🧠 Mental Health Chatbot")
    st.markdown("Ask anything related to mental health. The bot responds with relevant advice from experts.\n\n*Type 'exit' to clear the chat.*")

    # Language selection
    lang = st.selectbox("🌍 Choose your preferred language", ["en", "fr", "ar", "es", "de", "it", "zh", "ru", "pt"], index=0)

    # Load model and data
    questions, answers = load_data()
    model, question_embeddings = load_model_and_embeddings(questions)

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    user_input = st.text_input("👤 You:", key="user_input")
    if user_input:
        if user_input.lower() == "exit":
            st.session_state.history = []
            st.experimental_rerun()

        # Semantic search
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        response = "🤖 I'm sorry, I couldn't find a suitable answer."

        if best_score > 0.5:
            answer_en = answers[best_idx]
            if lang != "en":
                try:
                    translated = GoogleTranslator(source='en', target=lang).translate(answer_en)
                    response = f"🤖 {translated} _(translated from English)_"
                except Exception:
                    response = f"🤖 {answer_en} ⚠ (translation failed)"
            else:
                response = f"🤖 {answer_en}"

        # Save chat history
        st.session_state.history.append((user_input, response))

    # Display chat
    for user_q, bot_a in st.session_state.history:
        st.markdown(f"**👤 You:** {user_q}")
        st.markdown(bot_a)

if __name__ == "__main__":
    main()
