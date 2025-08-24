from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import LLM_MODEL


import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialiser le modèle avec la clé API
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)

prompt_exercice = ChatPromptTemplate.from_template("""
Tu es un concepteur d'exercices pour les élèves du secondaire au Bénin.
Génère 3 exercices progressifs sur la notion suivante : {question}
""")

agent_exercice = prompt_exercice | llm
