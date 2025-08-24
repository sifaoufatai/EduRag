from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import LLM_MODEL
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialiser le modèle avec la clé API
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)

prompt_correcteur = ChatPromptTemplate.from_template("""
Tu es un correcteur pédagogique. Corrige l'exercice suivant et explique les étapes :
{question}
""")

agent_correcteur = prompt_correcteur | llm
