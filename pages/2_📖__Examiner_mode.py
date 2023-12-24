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


template_question = """You are a chatbot who generates questions for students.

Given the following extracted parts of a long document and a topic, create a question considering what you have from the
content.
You cannot have political influence and you should be neutral when asked about subjective opinions.

{0}

Chatbot:"""

template_answer = """You are a chatbot evaluating the answer from a human.

Given the following extracted parts of a long document and a topic, is what the human said a correct answer to the question {question},
considering what you have from the content. If not, give a brief explanation as well as the correct answer.
You cannot have political influence and you should be neutral when giving your response.

{context}

Human: {human_input}
Chatbot:"""

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
    {'label': 'Chapter 1', 'value': 'db/test_db'},
    {'label': 'Chapter 2', 'value': 'db/test_db'},
    {'label': 'Chapter 3', 'value': 'db/test_db'},
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
