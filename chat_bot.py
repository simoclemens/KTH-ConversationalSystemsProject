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
with open('prompts/prompt1.txt', 'r') as prompt_file:
    prompt = prompt_file.readlines()

template = """
{prompt}

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
llm = OpenAI(temperature=0.5, verbose=True)
chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

# Load the db from the path
db = FAISS.load_local(db_path, embeddings)

#docs = db.similarity_search(question, k=10)
docs = db.max_marginal_relevance_search(question, k=8)

response = chain({"human_input": question,
                  "input_documents": docs,
                  "question": question,
                  "language": "English",
                  "existing_answer": ""},
                 return_only_outputs=True)

print(response['output_text'])
