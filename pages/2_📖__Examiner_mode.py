import os
import utils
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import MultiPromptChain
from streaming import StreamHandler
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import ConversationalRetrievalChain


os.environ["OPENAI_API_KEY"] = "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"
db_path = "db/test_db"

st.set_page_config(page_title="Examiner mode", page_icon='📖')

class ExaminerChatbot:
    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
        self.memory = None
    '''
    def run(self):
        st.button("New question", onClick=self.generateQuestionThread)
    '''
    def setup(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history")#return_messages=True)

        # Define the embedding function
        embeddings = OpenAIEmbeddings()

        # Load the db from the path
        db = FAISS.load_local(db_path, embeddings)
        
        # Define retriever
        retriever = db.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        with open('prompts/evaluation_prompt.txt', 'r') as prompt_file:
            eval_template = prompt_file.readlines()
        eval_prompt = eval_template[0] + ' The student gave this answer: {input}. Is it correct based on the information given in {context}? '
        

        with open('prompts/question_prompt.txt', 'r') as prompt_file2:
            question_template = prompt_file2.readlines()
        question_prompt = question_template[0]+' {context}. Give only one question.'

        eval_prompt = PromptTemplate.from_template(eval_template[0])
        question_prompt = PromptTemplate.from_template(question_template[0])

        prompt_names = ["question", "evaluate"]
        
        # TODO
        prompt_descriptions = ["The user wants to be asked a question", "The answer should be evaluated against the source"]

        prompt_templates = [question_prompt, eval_prompt]
        prompt_infos = [{"name": prompt_names[0], 
                         "description":prompt_descriptions[0],
                         "prompt": prompt_templates[0]}, 
                         {"name": prompt_names[1], 
                         "description":prompt_descriptions[1],
                         "prompt": prompt_templates[1]}
                        ] 

        llm = ChatOpenAI(model_name=self.openai_model, temperature = 0.3, streaming=True) # maybe change streaming
        
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt = p_info["prompt"]

            chain = ConversationalRetrievalChain.from_llm(llm,retriever=retriever,
            memory=self.memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt})

            destination_chains[name] = chain
        default_chain = ConversationChain(llm=llm, output_key="text")

        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)

        big_chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)
        
        return big_chain
    
    @utils.enable_chat_history # remove?
    def question(self):
        input = st.chat_input()
        chain = self.setup()

        if(input):
            utils.display_msg(input, "user")
            with st.chat_message("examiner"):
                st_cb = StreamHandler(st.empty())
                response = chain.run(input)#, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})



# Create a text input box for the user
#input = st.text_input('Give answer')
if __name__ == "__main__":
    obj = ExaminerChatbot()
    obj.question()