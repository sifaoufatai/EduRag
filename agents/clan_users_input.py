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
Tu es un expert en prompts pour modèles de langage.  
Ton objectif est de comprendre ce que l'élève essaie de dire et de reformuler sa question clairement pour le modèle.  

- L'élève est un collégien du Bénin.  
- Il pose des questions sur son programme scolaire, surtout en mathématiques.  

Instructions :  
1. Identifier exactement la question de l'élève.  
2. Reformuler la question clairement, sans inventer de réponse.  
3. **Conserver le niveau de l'élève**.  
4. Si le niveau n’est pas clair, **demande-le explicitement** avant de reformuler.  

""")

agent_exercice = prompt_exercice | llm
