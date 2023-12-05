from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = "XXX"

db_path = 'db/test_db'  # sys.argv[1]
question = "Quali sono le coperture?"  # sys.argv[2]

template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer. When you cannot find 
information in the context answer "Mi spiace, non ho trovato l'informazione rischiesta". You are allowed to answer 
only qestions related to insurance policies and yu need to decline any kind of other question aswering "Mi spiace ma 
non posso rispondere".

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history","human_input", "context"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

# Define the embedding function
embeddings = OpenAIEmbeddings()
# Define LLM model (default is a GPT3 davinci)
llm = OpenAI(temperature=0, verbose=True)
chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

# Load the db from the path
db = FAISS.load_local(db_path, embeddings)

#docs = db.similarity_search(question, k=10)
docs = db.max_marginal_relevance_search(question, k=8)

response = chain({"human_input": question,
                  "input_documents": docs,
                  "question": question,
                  "language": "Italian",
                  "existing_answer": ""},
                 return_only_outputs=True)

print(response['output_text'])
