import os
import utils
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate,  SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import MultiPromptChain
from streaming import StreamHandler
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import ConversationalRetrievalChain
from typing import Dict, Any, Set


class ConversationalRetrievalChainModified(ConversationalRetrievalChain):
    def __call__(self, input_str: str, **kwargs) -> Dict[str, Any]:
        outputs = super().__call__(input_str, **kwargs)
        return {'text': outputs['text']}  # Choose the appropriate key
    @property
    def output_keys(self) -> Set[str]:
        return {'text'}  # Choose the appropriate key

os.environ["OPENAI_API_KEY"] = "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"
db_path = "db/test_db"

st.set_page_config(page_title="Examiner mode", page_icon='📖')

class ExaminerChatbot:
    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
        self.memory = None

    def setup(self):
        print("\nin steup\n")

        # Define the embedding function
        embeddings = OpenAIEmbeddings()

        # Load the db from the path
        db = FAISS.load_local(db_path, embeddings)
        
        # Define retriever
        retriever = db.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


        

        with open('prompts/evaluation_prompt.txt', 'r') as prompt_file:
            eval_template = prompt_file.readlines()
        eval_prompt = eval_template[0] + ' The student gave this answer: {question}. Is it correct based on the information given in {context}? '
        

        with open('prompts/question_prompt.txt', 'r') as prompt_file2:
            question_template = prompt_file2.readlines()
        question_prompt = question_template[0]+' {context}. Give only one question.'

        general_system_template = r"""
        Is what the user said correct based on the following context\?
        ----
        {context}
        ----
        """
        general_system_template2 = r"""
        Given the below context, create one single question.
        ----
        {context}
        ----
        """

        question_prompt = SystemMessagePromptTemplate.from_template(general_system_template2)#question_template[0])
        eval_prompt = SystemMessagePromptTemplate.from_template(general_system_template)#eval_template[0])
        user_prompt = HumanMessagePromptTemplate.from_template("Answer: ```{question}```")

        question_p = ChatPromptTemplate.from_messages([question_prompt, user_prompt])
        eval_p = ChatPromptTemplate.from_messages([eval_prompt, user_prompt])

        prompt_names = ["question", "evaluate"]
        
        # TODO
        prompt_descriptions = ["The user wants to be asked a question", "The answer should be evaluated against the source"]

        prompt_templates = [question_p, eval_p]
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
                    
            chain = ConversationalRetrievalChainModified.from_llm(llm, retriever=retriever, memory=self.memory, verbose=True, combine_docs_chain_kwargs={"prompt": prompt})
        
            # Set the desired output_key for the chain
            chain.output_key = 'text'
            
            print(chain.input_keys)
            destination_chains[name] = chain
        default_chain = ConversationChain(llm=llm, output_key='text', input_key='input')
        print("DEF", default_chain.input_keys)


        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=['input'],
            output_parser=RouterOutputParser(next_inputs_inner_key='question'),
        )
      
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        print(router_chain.input_keys)

        big_chain = MultiPromptChain(memory=self.memory, router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)
        print(big_chain.input_keys) 
        return big_chain
    
    @utils.enable_chat_history # remove?
    def question(self):
        user_query = st.chat_input()
        chain = self.setup()
        print("back from setup")
        if user_query:
            utils.display_msg(user_query, "user")
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                print("about to run chain")
                response = chain.run(user_query, callbacks=[st_cb])
                print(response)
                st.session_state.messages.append({"role": "assistant", "content": response})



# Create a text input box for the user
#input = st.text_input('Give answer')
if __name__ == "__main__":
    bot = ExaminerChatbot()
    bot.question()