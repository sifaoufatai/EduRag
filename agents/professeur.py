from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from config import LLM_MODEL


import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialiser le modèle avec la clé API
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
# Prompt du professeur
prompt_professeur = ChatPromptTemplate.from_template("""
Tu es un professeur de mathématiques du secondaire au Bénin.
Explique la notion suivante clairement, avec des exemples adaptés au niveau de l'élève :
{question}
""")

# Chaîne de l'agent
agent_professeur = prompt_professeur | llm
