import sys
import os
import pysqlite3

# Rediriger sqlite3 pour utiliser pysqlite3 (compatibilité avec ChromaDB)
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import (
    ChatOpenAI,
    ChatAnthropic,
    ChatCohere,
)

try:
    from langchain_mistralai.chat_models import ChatMistralAI
except ImportError:
    ChatMistralAI = None

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ------------------ CONFIG ------------------ #
VECTOR_DB_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_OPTIONS = {
    "OpenAI (gpt-3.5-turbo)": ("openai", "gpt-3.5-turbo"),
    "Anthropic (Claude 3 Opus)": ("anthropic", "claude-3-opus-20240229"),
    "Mistral (Tiny)": ("mistral", "mistral-tiny"),
    "Cohere (Command R+)": ("cohere", "command-r-plus"),
}

# ------------------ UI ------------------ #
st.title("Assistant éducatif du Bénin")

llm_label = st.sidebar.selectbox("Choisissez votre modèle LLM", list(LLM_OPTIONS.keys()))
provider, model_name = LLM_OPTIONS[llm_label]
api_key = st.sidebar.text_input("Clé API pour le modèle sélectionné", type="password")

# ------------------ HELPERS ------------------ #
def check_model_change(selected_model: str):
    """Réinitialise l’historique si l’utilisateur change de modèle."""
    if "last_model" not in st.session_state:
        st.session_state.last_model = selected_model
    elif selected_model != st.session_state.last_model:
        st.session_state.chat_history = []
        st.session_state.last_model = selected_model

check_model_change(llm_label)

@st.cache_resource
def load_embeddings():
    """Charge le modèle d’embeddings HuggingFace."""
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Erreur lors du chargement des embeddings : {e}")
        st.stop()

@st.cache_resource
def load_vector_db():
    """Charge la base vectorielle persistante Chroma."""
    try:
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=load_embeddings()
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement de la base vectorielle : {e}")
        st.stop()

def initialize_memory():
    """Initialise la mémoire de conversation."""
    try:
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    except Exception as e:
        st.error(f"Erreur lors de l’initialisation de la mémoire : {e}")
        st.stop()

def load_llm(provider: str, model_name: str, api_key: str):
    """Charge le LLM choisi avec sa clé API."""
    try:
        if provider == "openai":
            return ChatOpenAI(model_name=model_name, openai_api_key=api_key)
        elif provider == "anthropic":
            return ChatAnthropic(model=model_name, anthropic_api_key=api_key)
        elif provider == "mistral":
            if ChatMistralAI is None:
                raise ImportError("`ChatMistralAI` nécessite `langchain-mistralai`. Installez-le avec : pip install langchain-mistralai")
            return ChatMistralAI(model=model_name, mistral_api_key=api_key)
        elif provider == "cohere":
            return ChatCohere(model=model_name, cohere_api_key=api_key)
        else:
            raise ValueError(f"Fournisseur inconnu : {provider}")
    except Exception as e:
        st.error(f"Erreur lors du chargement du LLM : {e}")
        st.stop()

def get_user_input():
    """Récupère la question de l’utilisateur et la nettoie après envoi."""
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""
    st.text_input("Posez votre question :", key="user_question", on_change=save_and_clear)
    return st.session_state["last_question"]

def save_and_clear():
    """Sauvegarde la question de l’utilisateur et vide le champ."""
    st.session_state["last_question"] = st.session_state["user_question"]
    st.session_state["user_question"] = ""

def clean_text(text: str) -> str:
    """Supprime les retours à la ligne du texte."""
    return text.replace("\n", " ")

# ------------------ INIT ------------------ #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not api_key or api_key.strip() == "":
    st.error("Veuillez entrer une clé API valide pour le modèle sélectionné.")
    st.stop()

llm = load_llm(provider, model_name, api_key)
vector_db = load_vector_db()
memory = initialize_memory()

# ------------------ PROMPT ------------------ #
custom_prompt = PromptTemplate(
    template="""
Tu es un **assistant éducatif du Bénin**.

Ton comportement dépend de la question de l’utilisateur :

1. Si c’est une salutation → Réponds chaleureusement et invite l’utilisateur à indiquer sa classe.  
   ⚠️ Les classes au Bénin sont appelées **grade** dans les métadonnées du contexte :  
   (6e, 5e, 4e, 3e, 2nde, 1ereA1, 1ereA2, 1ereC, 1ereD, 2ndeC, 2ndeD, TleC).  
   Exemple : si un élève dit « je suis en 6ème », tu dois récupérer les contenus correspondant au grade **6e**.

2. Une fois le grade identifié → Demande-lui quelle notion il souhaite étudier.

3. Recherche la notion dans la classe (grade) indiquée :  
   - Si elle existe → fournis les explications adaptées.  
   - Si elle n’existe pas dans le grade indiqué → cherche dans les grades inférieurs.  
   - Si elle n’existe que dans les grades supérieurs → informe l’élève que cela dépasse son niveau actuel.

4. Si la question est **juridique** → Répond uniquement à partir du **contexte fourni** et cite les **articles de loi** correspondants.

5. Si aucun contexte n’est trouvé → Indique clairement à l’utilisateur qu’aucune information n’est disponible.

⚠️ Règles importantes :  
- Toujours répondre **en français**.  
- Ne jamais inventer d’informations.  

---

Historique de conversation :  
{chat_history}

Question utilisateur :  
{question}

Contexte (extraits pertinents du contenu ou du Code du Travail) :  
{context}

Réponse :
""",
    input_variables=["question", "context", "chat_history"]
)

try:
    custom_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
except Exception as e:
    st.error(f"Erreur lors de la création de la chaîne de conversation : {e}")
    st.stop()

# ------------------ INTERACTIONS ------------------ #
def handle_user_interaction():
    """Gère l’interaction utilisateur et affiche la réponse + contexte."""
    user_input = get_user_input()
    if user_input:
        cleaned_question = clean_text(user_input)
        try:
            retriever = vector_db.as_retriever()
            relevant_docs = retriever.get_relevant_documents(cleaned_question)

            result = custom_chain({
                "question": cleaned_question,
                "chat_history": st.session_state.chat_history
            })

            st.session_state.chat_history.append((cleaned_question, result["answer"]))
            st.write(result["answer"])

            if relevant_docs:
                with st.expander("Afficher le contexte utilisé et les références"):
                    for doc in relevant_docs:
                        st.write(doc.page_content)
                        st.write("**Références :**")
                        for key, value in doc.metadata.items():
                            st.write(f"{key}: {value}")
                        st.write("---" * 40)
            else:
                st.info("Aucun contexte trouvé.")

        except Exception as e:
            st.error(f"Erreur lors de la génération de la réponse : {e}")

def display_chat_history_sidebar():
    """Affiche l’historique de la conversation dans la sidebar."""
    with st.sidebar:
        if st.session_state.chat_history:
            with st.expander("Historique de conversation :"):
                for q, a in st.session_state.chat_history:
                    st.markdown(f"**Vous :** {q}")
                    st.markdown(f"**Assistant :** {a}")

# ------------------ MAIN ------------------ #
handle_user_interaction()
display_chat_history_sidebar()
