import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="TF-IDF Galáctico", page_icon="🌌", layout="wide")

# --- ESTILO GALÁCTICO ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #0a0024, #000014, #020024);
            color: #e0e0ff;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12003a, #23006b);
            color: #ffffff;
        }
        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
        h1, h2, h3, h4 {
            color: #b49aff !important;
            text-shadow: 0 0 10px #7f5af0;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a00f4, #b517ff);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.6em 1.4em;
            font-weight: bold;
            box-shadow: 0 0 16px #8e2de2;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #b517ff, #6a00f4);
            transform: scale(1.05);
            box-shadow: 0 0 22px #a29bfe;
        }
        .stTextInput>div>div>input,
        .stTextArea>div>textarea {
            background-color: rgba(25, 10, 50, 0.7);
            color: #e0e0ff;
            border-radius: 8px;
            border: 1px solid #6a00f4;
        }
        .stDataFrame {
            background-color: rgba(15, 0, 40, 0.6);
            border-radius: 12px;
        }
        .stMarkdown p {
            color: #e0d9ff;
        }
        .stAlert {
            background-color: rgba(35, 0, 90, 0.3);
            border: 1px solid #7f5af0;
        }
    </style>
""", unsafe_allow_html=True)

# --- CONTENIDO PRINCIPAL ---
st.title("🪐 TF-IDF Galáctico: Explorando palabras en el universo del texto 🌠")

st.write("""
Cada línea es tratada como un **documento** (una frase, un párrafo o incluso un texto completo).  
🛰️ Los documentos y las preguntas deben estar en **inglés**, ya que el modelo está configurado para ese idioma.  
Las palabras son procesadas con *stemming*, para que palabras como *playing* y *play* se consideren iguales. ✨
""")

# Ejemplo inicial
text_input = st.text_area(
    "📜 Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("💫 Escribe una pregunta (en inglés):", "Who is playing?")

# Inicializar stemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems


if st.button("🚀 Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento para despegar 🚀")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)

        # Mostrar matriz TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.write("### 🌌 Matriz TF-IDF (stems detectados)")
        st.dataframe(df_tfidf.round(3))

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])

        # Similitud coseno
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más parecido
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 🧠 Análisis de coincidencias cósmicas")
        st.markdown(f"**Pregunta:** {question}")
        st.markdown(f"**Documento más relevante:** 🌟 *Doc {best_idx+1}* → `{best_doc}`")
        st.markdown(f"**Puntaje de similitud:** `{best_score:.3f}`")

        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })

        st.write("### 🔭 Similitudes detectadas entre tus documentos:")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False))

        # Mostrar coincidencias de stems
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.write("### 🌠 Stems de la pregunta presentes en el documento elegido:")
        st.write(matched)
