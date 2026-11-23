from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def load_and_chunk_documents(folder_path):
    documents = []

    print(f"Loading documents from: {folder_path}")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = os.listdir(folder_path)
    print("Files found:", files)

    for file in files:
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            print(f"Loading PDF: {file_path}")

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            documents.extend(docs)

            print(f"Loaded {len(docs)} pages from {file}")

    print(f"Total pages loaded: {len(documents)}")

    if len(documents) == 0:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks