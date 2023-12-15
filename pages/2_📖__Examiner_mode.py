import os
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"


def get_answer(input, chain, db):
    docs = db.similarity_search(input, k=10)
    # docs = db.max_marginal_relevance_search(question, k=8)

    response = chain({"human_input": input,
                      "input_documents": docs,
                      "language": "English",
                      "existing_answer": ""},
                     return_only_outputs=True)
    return response['output_text']

def get_answer2(input, chain, db, question):
    docs = db.similarity_search(input, k=10)
    # docs = db.max_marginal_relevance_search(question, k=8)

    response = chain({"human_input": input,
                      "input_documents": docs,
                      "language": "English",
                      "existing_answer": "", 
                      "question": question},
                     return_only_outputs=True)
    return response['output_text']


db_path = 'db/test_db'  # sys.argv[1]

template_question = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a topic, create a question for the user about the specific topic
considering what you have from the content
You cannot have political influence and you should be neutral when asked about subjective opinions.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

template_answer = """You are a chatbot evaluating the answer from a human.

Given the following extracted parts of a long document and a topic, is what the human said a correct answer to the question {question},
considering what you have from the content. If not, give a brief explanation as well as the correct answer.
You cannot have political influence and you should be neutral when giving your response.

{context}

Human: {human_input}
Chatbot:"""

prompt_question = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template_question
)

prompt_answer = PromptTemplate(
    input_variables=["chat_history", "human_input", "context", "question"], template=template_answer
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

# Define the embedding function
embeddings = OpenAIEmbeddings()
# Define LLM model (default is a GPT3 davinci)
llm = OpenAI(temperature=0.5, verbose=True)

chain_question = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt_question)

chain_eval = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt_answer)

# Load the db from the path
db = FAISS.load_local(db_path, embeddings)

st.title('Questions time!📖')
if 'question_generated' not in st.session_state:
    st.session_state['question_generated'] = False
if 'question' not in st.session_state:
    st.session_state['question'] = ""


option = st.selectbox(
    'Select the chapter',
    ('Chapter 1', 'Chapter 2', 'Chapter 3'))

# Create a text input box for the user
input_topic = st.text_input('Tell me a topic')
generate_button = st.button("Generate question")
new_question = False
# If the user hits enter
if input_topic and not st.session_state['question_generated']:
    response_q = get_answer(input_topic, chain_question, db)
    st.write(response_q)
    st.session_state['question_generated'] = True
    st.session_state['question'] = response_q
    new_question = True
    

# Create a text input box for the user

input_ans = st.text_input('Give your answer')


# If the user hits enter
if input_ans and not new_question:
    response_e = get_answer2(input_ans, chain_eval, db, st.session_state['question'])
    st.write(response_e)
    st.session_state['question_generated'] = False
    



