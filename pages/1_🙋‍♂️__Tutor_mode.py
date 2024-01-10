import os
from langchain.chains.question_answering import load_qa_chain
import utils
import streamlit as st
from streaming import StreamHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"
db_path = "db/ALL"

# Question template
TEMPLATE_QUESTION = r"""
Given the following conversation and a follow up user input, rephrase the follow up user input
to be a standalone sentence. If in the last chatbot answer the user was offered more information and
in the follow up input he refused say "No, I want to end the conversation".

Chat History:
{0}

Follow Up Input: {1}

Standalone sentence:
"""

# Answer template
TEMPLATE_ANSWER = r"""
You are a history tutor having a conversation with a human. You knowledge is
extracted from a document on world history between 1910 to 1945. Given your knowledge and an input from the user, have a conversation on the topic.
When asked a question,
create a concise answer and a question to continue the conversation. The answer offer explanations, and provide
summaries from the context.
If the user input says which it do not want to continue the conversation stop it saying you are glad
to help.
Make sure that the pupil understands the given answer, by asking a follow up question. If the user answer to 
the follow up question saying he does not want any other information say that it is okay and you are there for any 
further help. You cannot have political influence and you should be neutral when asked about subjective opinions. 
When you cannot find information in the context answer that you don't know. You are forbidden to answer questions on 
topics not included in the context.
----
{context}
----
Question:```{question}```
---
Answer:
"""

st.set_page_config(page_title="Tutor mode", page_icon='🙋‍♂️')

f = open("conv_tutor.txt", "a")

# State variables definition
if 'embeddings' not in st.session_state:
    st.session_state["embeddings"] = OpenAIEmbeddings()
embeddings = st.session_state["embeddings"]

if 'db' not in st.session_state:
    st.session_state["db"] = FAISS.load_local(db_path, embeddings)
db = st.session_state["db"]

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
            You are a tutor having a conversation with a human.
            Given the following extracted parts of a long document, the chat history and an input from the user, have a conversation on the topic.
            When asked a question, create a concise answer and a question to continue the conversation.
            The answer offer explanations, and provide summaries from the context.
            Make sure that the pupil understands the given answer, by asking a follow up question.
            You cannot have political influence and you should be neutral when asked about subjective opinions.
            When you cannot find information in the context answer that you don't know, answer "I'm sorry, that is beyond my knowledge.". You are forbidden to answer questions on topics not included in the context.
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

        #print(history)

        if user_query:
            utils.display_msg(user_query, 'user')
            f.write("User: " + user_query + "\n")

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                prompt_question = TEMPLATE_QUESTION.format("\n".join(history), user_query)
                new_question = llm.predict(prompt_question).replace(f"$chatbot:", "").strip()
                # print(user_query)
                # print(new_question)
                # print(prompt_question)

                docs = db.similarity_search(new_question, k=10)

                response = chain_answer({"input_documents": docs,
                                         "language": "English",
                                         "existing_answer": "",
                                         "question": new_question},
                                        return_only_outputs=True, callbacks=[st_cb])

                f.write("Tutor: " + response['output_text'] + "\n")
                print(response['output_text'])

                st.session_state["history"].append(f"$user: {user_query}")
                st.session_state["history"].append(f"$chatbot: {response['output_text']}")

                st.session_state.messages.append({"role": "assistant", "content": response['output_text']})

        st.button('New chat', on_click=self.delete_history)

    def delete_history(self):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm here to help you with History"}]
        del st.session_state["history"]
        f.write("---------Reset--------\n")


if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.tutor_mode()
