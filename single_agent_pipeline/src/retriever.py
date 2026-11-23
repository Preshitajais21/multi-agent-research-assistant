from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_PATH = os.path.join(BASE_DIR, "vector_store", "faiss_index")

os.makedirs(VECTOR_PATH, exist_ok=True)


def create_vector_store(chunks):
    print("Creating vector store...")

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_PATH)

    print(f"Vector store saved at: {VECTOR_PATH}")


def load_vector_store():
    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


def retrieve_context(query, k=4):
    db = load_vector_store()

    docs = db.similarity_search(query, k=k)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context