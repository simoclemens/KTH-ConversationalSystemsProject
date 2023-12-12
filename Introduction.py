import streamlit as st

st.set_page_config(
    page_title="Educational Chatbot",
    page_icon='💬',
    layout='wide'
)

st.title("EduChat 💬")
st.header("Helping students studying in a fast and safe way")

st.write("This project aims to provide students with an AI assistant specifically modified for educational tasks."
         "The chatbot is based on a gpt-3.5-turbo model which was prompted in order to give safer answers anf to get "
         "information only from sources given as input: the idea is that this product can become the ideal textbook "
         "companion")

st.write("Project developed for the course DT2151 Project in Conversational Systems, KTH")
st.write("Authors: Simone Clemente, Elin Saks, Tora Wallerö")