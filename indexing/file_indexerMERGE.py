import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = "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"

db_path = 'db/ALL'

# Define the embedding function
embeddings = OpenAIEmbeddings()


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)


pdfs = ["documents/chap23.pdf","documents/chap24.pdf", "documents/chap25.pdf", "documents/chap26.pdf"]

for index, pdf in enumerate(pdfs):
    # Create and load PDF Loader
    loader = PyPDFLoader(pdf)
    docs = loader.load_and_split(text_splitter=text_splitter)
    if index == 0:
        faiss_index = FAISS.from_documents(docs, embeddings)
    else:
        faiss_index_i = FAISS.from_documents(docs, embeddings)
        faiss_index.merge_from(faiss_index_i)

faiss_index.save_local(db_path)
