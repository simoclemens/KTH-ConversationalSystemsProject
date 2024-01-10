import os

from langchain.chains.question_answering import load_qa_chain

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

TEMPLATE_QUESTION = r"""
Given the following conversation and a follow up question, rephrase the follow up question
to be a standalone question.

Follow Up Input: {0}

Chat History:
{1}

Standalone question:
"""

TEMPLATE_ANSWER = r"""
    You are a tutor having a conversation with a human.
    Given the following extracted parts of a long document and an input from the used, have a conversation on the topic.
    When asked a question, create an concise answer and a question to continue the conversation.
    The answer offer explanations, and provide summaries from the context.
    Make sure that the pupil understand the given answer, by asking a follow up question.
    You cannot have political influence and you should be neutral when asked about subjective opinions.
    When you cannot find information in the context answer that you don't know, answer "I'm sorry, that is beyond
    my knowledge.". You are forbidden to answer questions on topics not included in the context.
----
{context}
----
Question:```{question}```
"""

os.environ["OPENAI_API_KEY"] = "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"
db_path = "db/ALL"

st.set_page_config(page_title="Tutor mode", page_icon='🙋‍♂️')

f = open("conv_tutor.txt", "a")

# f.write("---------New conversation--------\n")

if 'embeddings' not in st.session_state:
    st.session_state["embeddings"] = OpenAIEmbeddings()
embeddings = st.session_state["embeddings"]

if 'db' not in st.session_state:
    st.session_state["db"] = FAISS.load_local(db_path, embeddings)
db = st.session_state["db"]

if 'prompt_question' not in st.session_state:
    st.session_state["prompt_question"] = PromptTemplate(
        input_variables=["chat_history", "question"], template=TEMPLATE_QUESTION
    )
prompt_question = st.session_state["prompt_question"]

if 'prompt_answer' not in st.session_state:
    st.session_state["prompt_answer"] = PromptTemplate(
        input_variables=["context", "question"], template=TEMPLATE_ANSWER
    )
prompt_answer = st.session_state["prompt_answer"]

if 'llm' not in st.session_state:
    st.session_state['llm'] = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True)
llm = st.session_state['llm']

if 'history' not in st.session_state:
    st.session_state["history"] = []
    st.session_state["history"].append(f"$chatbot: Hi! I'm here to help you with History")
history = st.session_state['history']

if 'chain_answer' not in st.session_state:
    st.session_state["chain_answer"] = load_qa_chain(llm, chain_type="stuff", prompt=prompt_answer)
chain_answer = st.session_state["chain_answer"]


class CustomDataChatbot:

    def __init__(self):
        pass

    @utils.enable_chat_history
    def tutor_mode(self):
        user_query = st.chat_input(placeholder="Ask me anything!")

        print(history)

        if user_query:
            utils.display_msg(user_query, 'user')
            f.write("User: " + user_query + "\n")

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                prompt_question = TEMPLATE_QUESTION.format(user_query, "\n".join(history))
                new_question = llm.predict(prompt_question).replace(f"$chatbot:", "").strip()
                print(user_query)
                print(new_question)

                print(prompt_question)

                db = FAISS.load_local(db_path, embeddings)
                docs = db.similarity_search(new_question, k=10)

                response = chain_answer({"human_input": input,
                                         "input_documents": docs,
                                         "language": "English",
                                         "existing_answer": "",
                                         "question": new_question},
                                        return_only_outputs=True, callbacks=[st_cb])
                f.write("Tutor: " + response['output_text'] + "\n")
                st.session_state["history"].append(f"$user: {user_query}")

                st.session_state["history"].append(f"$chatbot: {response['output_text']}")

                st.session_state.messages.append({"role": "assistant", "content": response})

        st.button('New chat', on_click=self.delete_history)

    def delete_history(self):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm here to help you with History"}]
        del st.session_state["history"]
        f.write("---------Reset--------\n")


if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.tutor_mode()
