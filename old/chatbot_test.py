import os
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import openai

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
llm = ChatOpenAI(temperature=0.5, verbose=True)

chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

# Load the db from the path
db = FAISS.load_local(db_path, embeddings)

col1, col2 = st.columns(2)

with col1:
    option = st.selectbox(
        'Select the chapter',
        ('Chapter 1', 'Chapter 2', 'Chapter 3'))

with col2:
    examiner_mode = st.toggle('Examiner mode')

if not examiner_mode:

    st.title('Hi, I am your tutor!🙋‍♂️')

    client = OpenAI(model="gpt-3.5-turbo")

    st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})



else:

    st.title('Questions time!📖')
    # Create a text input box for the user
    input = st.text_input('Tell me a topic')

    # If the user hits enter
    if input:
        response = get_answer(input, chain, db)
        st.write(response)
