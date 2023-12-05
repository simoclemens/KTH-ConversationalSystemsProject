import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = "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"

file_path = 'documents/test.pdf'
db_path = 'db/test_db'

# Define the embedding function
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader(file_path)
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)
# Create a list of documents splitting the PDF
docs = loader.load_and_split(text_splitter=text_splitter)
# Generate the embedding database
db = FAISS.from_documents(docs, embeddings)
# Save the db
db.save_local(db_path)
