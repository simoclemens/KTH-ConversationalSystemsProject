import random

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import os

os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()
db_path = "db/test_db"
db = FAISS.load_local(db_path, embeddings)

input = ""
n_docs = db.index.ntotal
docs = db.similarity_search(input, k=n_docs)
random_ind = random.sample(range(n_docs), 5)
selected_docs = [docs[i] for i in random_ind]
print(type(docs))



