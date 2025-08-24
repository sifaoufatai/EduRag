from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.professeur import agent_professeur
from agents.exercice import agent_exercice
from agents.correcteur import agent_correcteur

# Définition du schéma de l'état
class ChatState(TypedDict):
    question: str

# Fonction de routage (Tuteur)
def route_request(state: ChatState):
    question = state["question"].lower()
    if "exercice" in question:
        return {"next_node": "exercice"}       # retourne un dict
    elif "corrige" in question or "réponse" in question:
        return {"next_node": "correcteur"}     # retourne un dict
    else:
        return {"next_node": "professeur"}     # retourne un dict

# Création du graphe avec le schéma d'état
graph = StateGraph(ChatState)

# Ajouter les agents comme nodes
graph.add_node("professeur", agent_professeur)
graph.add_node("exercice", agent_exercice)
graph.add_node("correcteur", agent_correcteur)

# Ajouter le routeur comme node
graph.add_node("router", route_request)

# Définir le point d'entrée sur le routeur
graph.set_entry_point("router")

# Définir les sorties vers END pour chaque agent
graph.add_edge("professeur", END)
graph.add_edge("exercice", END)
graph.add_edge("correcteur", END)

# Compiler le graphe
compiled_graph = graph.compile()
