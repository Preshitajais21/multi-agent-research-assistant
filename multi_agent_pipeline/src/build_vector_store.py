import re
from pathlib import Path

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def clean_text(text: str) -> str:
    text = re.sub(r"Cite this version:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_filename_title(filename: str) -> str:
    title = filename.replace(".pdf", "")
    title = title.replace("_", " ").replace("-", " ").replace("+", " ")
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def get_pdf_title(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))

        # 1. Try metadata first
        metadata = reader.metadata
        if metadata and metadata.title:
            title = metadata.title.strip()
            if len(title) > 5:
                return title

        # 2. Fallback: extract from first page
        first_page = reader.pages[0].extract_text()

        if first_page:
            lines = first_page.split("\n")

            # take first meaningful long line as title
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not line.lower().startswith(("abstract", "introduction")):
                    return line

    except Exception:
        pass

    # 3. Final fallback → cleaned filename
    return clean_filename_title(pdf_path.name)


def load_pdfs(data_folder: str):
    documents = []
    pdf_files = list(Path(data_folder).glob("*.pdf"))

    for pdf_file in pdf_files:
        paper_title = get_pdf_title(pdf_file)

        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = pdf_file.name
            doc.metadata["title"] = paper_title

        documents.extend(docs)

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return text_splitter.split_documents(documents)


def build_and_save_vector_store(data_folder="data", save_path="faiss_index"):
    print("Loading PDF documents...")
    documents = load_pdfs(data_folder)
    print(f"Loaded {len(documents)} pages.")

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    print(f"Saving vector store to '{save_path}'...")
    vector_store.save_local(save_path)

    print("Done! Vector store is ready.")


if __name__ == "__main__":
    build_and_save_vector_store()