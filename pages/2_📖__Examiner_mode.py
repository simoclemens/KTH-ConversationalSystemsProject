import os
import random
import re

import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"

def get_question(model, db_path, embeddings):

    db = FAISS.load_local(db_path, embeddings)

    input = ""
    n_docs = db.index.ntotal
    docs = db.similarity_search(input, k=n_docs)
    random_ind = random.sample(range(n_docs), 5)
    selected_docs = [docs[i] for i in random_ind]
    information = "\n\n".join([re.sub(r'\s+', ' ', doc.page_content) for doc in selected_docs])
    prompt = template_question.format(information)
    question = model.predict(prompt).replace(f"$chatbot:", "").strip()
    return question


def get_eval(input, chain, db_path, question, embeddings):
    db = FAISS.load_local(db_path, embeddings)
    docs = db.similarity_search(input, k=10)
    # docs = db.max_marginal_relevance_search(question, k=8)

    response = chain({"human_input": input,
                      "input_documents": docs,
                      "language": "English",
                      "existing_answer": "",
                      "question": question},
                     return_only_outputs=True)

    return response['output_text']

# CHANGE!!
db_path = 'db/ALL'  # sys.argv[1]


template_question = """You are a teacher who will enhance my knowledge through quizzes.
    You will teach by posing questions on a subject of my choice. 
    Please create an open-ended question for me based on the following document, do not refer to any images or tables present in the document. 
    Do not mention that your question is based on the document.
    Please also provide some context from the document before asking the question so that the user understands where the question is coming from.

    {0}

    Chatbot:"""

template_answer = """
    You are a teacher who will enhance the user's knowledge through quizzes.
    You will facilitate their learning by offering hints, clues, and suggestions for clearer explanations when the user struggle to answer fully.
    
    The question you gave the user was: {question}
    User answer: {human_input}

    Please evaluate the answer by comparing it to the information in the following document: {context}
    """

prompt_question = PromptTemplate(
    input_variables=["context"], template=template_question
)

prompt_answer = PromptTemplate(
    input_variables=["chat_history", "human_input", "context", "question"], template=template_answer
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

# Define the embedding function
embeddings = OpenAIEmbeddings()

question_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

# Define LLM model (default is a GPT3 davinci)
llm = OpenAI(temperature=0.5, verbose=True)

chain_eval = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt_answer)

st.title('Questions time!📖')
if 'question_generated' not in st.session_state:
    st.session_state['question_generated'] = False
if 'question' not in st.session_state:
    st.session_state['question'] = ""

chapters = [
    {'label': 'The Twentieth-Century Crisis', 'value': 'db/ch23_db'},
    {'label': 'The West Between the Wars', 'value': 'db/ch24_db'},
    {'label': 'Nationalism Around the World', 'value': 'db/ch25_db'},
    {'label': 'Chapter 26', 'value': 'db/ch26_db'}
]

option = st.selectbox(label="Select chapter", options=chapters, format_func=lambda item: item['label'])

new_question = False

# Create a text input box for the user
generate_button = st.button("Generate question")
if generate_button:
    question = get_question(question_model, option['value'], embeddings)
    st.session_state['question_generated'] = True
    st.session_state['question'] = question
    new_question = True

with st.form("Give your answer",clear_on_submit=True):
    st.write(st.session_state['question'])
    input_ans = st.text_input("Answer")
    submitted = st.form_submit_button("Submit")

    # If the user hits enter
    if submitted and not new_question:
        evaluation = get_eval(input_ans, chain_eval, option['value'], st.session_state['question'], embeddings)
        st.write(evaluation)
        st.session_state['question_generated'] = False
