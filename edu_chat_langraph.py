import streamlit as st
from graph import compiled_graph

st.title("Assistant éducatif du Bénin")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_user_input():
    return st.text_input("Posez votre question :", key="user_input")

def handle_user_interaction():
    user_input = get_user_input()
    if user_input:
        result = compiled_graph.invoke({"question": user_input})
        st.session_state.chat_history.append((user_input, result["output"]))
        st.write(result["output"])

def display_chat_history():
    with st.sidebar:
        if st.session_state.chat_history:
            st.write("### Historique de conversation")
            for q, a in st.session_state.chat_history:
                st.markdown(f"**Vous :** {q}")
                st.markdown(f"**Assistant :** {a}")

handle_user_interaction()
display_chat_history()
