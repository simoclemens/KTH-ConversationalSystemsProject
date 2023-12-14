import os

import utils
import streamlit as st
from streaming import StreamHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = "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"
db_path = "db/test_db"

st.set_page_config(page_title="Tutor mode", page_icon='🙋‍♂️')


class CustomDataChatbot:

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
        self.memory = None

    def setup_qa_chain(self):
        # Define the embedding function
        embeddings = OpenAIEmbeddings()

        db = FAISS.load_local(db_path, embeddings)

        # Define retriever
        retriever = db.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        # Setup memory for contextual conversation
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        general_system_template = r"""
        Given the following context taken from a study book answer the question posed by the user.
        If the information is out of the context answer "I am sorry, it seems that I cannot help you here!".
        Answer only to questions related to the topic of the context.
        You should have a neutral political opinion.
        ----
        {context}
        ----
        """
        general_user_template = "Question:```{question}```"
        messages = \
            [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
            ]

        prompt = ChatPromptTemplate.from_messages(messages)

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True)

        qa_chain = ConversationalRetrievalChain.from_llm(llm,
                                                         retriever=retriever,
                                                         memory=self.memory,
                                                         verbose=True,
                                                         combine_docs_chain_kwargs={"prompt": prompt})
        return qa_chain

    @utils.enable_chat_history
    def tutor_mode(self):
        user_query = st.chat_input(placeholder="Ask me anything!")
        qa_chain = self.setup_qa_chain()

        if user_query:
            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

        st.button('New chat', on_click=self.delete_history)

    def delete_history(self):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
        self.memory.clear()

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.tutor_mode()
