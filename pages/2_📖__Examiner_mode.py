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
                      "question": question,
                      "language": "English",
                      "existing_answer": ""},
                     return_only_outputs=True)
    return response['output_text']


db_path = 'db/test_db'  # sys.argv[1]
question = "Cuban missile crisis"  # sys.argv[2]

template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a topic, create a question for the user about the specific topic
considering what you have from the content
You cannot have political influence and you should be neutral when asked about subjective opinions.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

# Define the embedding function
embeddings = OpenAIEmbeddings()
# Define LLM model (default is a GPT3 davinci)
llm = OpenAI(temperature=0.5, verbose=True)

chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

# Load the db from the path
db = FAISS.load_local(db_path, embeddings)

st.title('Questions time!📖')
# Create a text input box for the user
input = st.text_input('Tell me a topic')

# If the user hits enter
if input:
    response = get_answer(input, chain, db)
    st.write(response)
