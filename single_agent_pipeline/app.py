import streamlit as st
import os

from src.loader import load_and_chunk_documents
from src.retriever import create_vector_store, retrieve_context
from src.agent import literature_review_agent


st.set_page_config(page_title="Single Agent Literature Review")
st.title("📚 Single Agent Literature Review System")


# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_PATH = os.path.join(BASE_DIR, "data")
VECTOR_INDEX = os.path.join(
    BASE_DIR,
    "vector_store",
    "faiss_index",
    "index.faiss"
)


st.sidebar.header("Setup")

if st.sidebar.button("Build Knowledge Base"):
    try:
        with st.spinner("Building knowledge base..."):
            chunks = load_and_chunk_documents(PAPER_PATH)

            if len(chunks) == 0:
                st.error("No PDF content found. Check data/papers folder.")
                st.stop()

            create_vector_store(chunks)

        st.sidebar.success("Knowledge base built successfully!")

    except Exception as e:
        st.error(f"Error building knowledge base: {e}")


query = st.text_input(
    "Enter research question:",
    placeholder="Example: AI in healthcare diagnosis"
)


if st.button("Generate Review"):
    if not os.path.exists(VECTOR_INDEX):
        st.error("Please click 'Build Knowledge Base' first.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a research question.")
        st.stop()

    try:
        with st.spinner("Retrieving context..."):
            context = retrieve_context(query)

        with st.expander("Retrieved Context"):
            st.write(context)

        with st.spinner("Generating literature review..."):
            result = literature_review_agent(query, context)

        st.markdown(result)

        st.download_button(
            "Download Review",
            result,
            file_name="literature_review.txt"
        )

    except Exception as e:
        st.error(f"Error generating review: {e}")