# Multi-Agent Research Assistant for Literature Review (RAG)

This project compares two approaches to building an LLM research assistant over a corpus of academic PDFs using Retrieval-Augmented Generation (RAG): a **single-agent baseline** and a **multi-agent pipeline** with planning, critique/refinement, and LLM-as-a-judge evaluation.

## Why two pipelines?

The single-agent pipeline retrieves relevant chunks and answers directly. The multi-agent pipeline instead:

1. **Plans** the query into sub-questions,
2. Generates **two candidate answers** independently,
3. Has a **Judge Agent** select (or synthesize) the best final answer,
4. Evaluates the result with **LLM-as-a-Judge** and **RAGAS** metrics (faithfulness, relevance, etc.) against the baseline.

The goal was to measure whether the added structure (planning, multiple candidates, critique) produces measurably better literature-review answers than a straightforward single-agent RAG system — not just to build a more complex pipeline for its own sake.

## Repository Structure

```text
.
├── single_agent_pipeline/
│   ├── app.py              # Streamlit UI for the baseline
│   └── src/
│       ├── agent.py        # Single-agent RAG logic
│       ├── loader.py       # PDF loading/chunking
│       └── retriever.py    # Vector retrieval
│
├── multi_agent_pipeline/
│   ├── app.py              # Streamlit UI for the multi-agent system
│   └── src/
│       ├── build_vector_store.py  # Builds the FAISS index from data/*.pdf
│       ├── full_pipeline.py       # Planner → candidates → judge → evaluation
│       └── ragas_evaluation.py    # RAGAS metric evaluation
│
├── requirements.txt
└── Multi-Agent_Research_Assistant_Report.docx   # Full write-up & comparison results
```

Each pipeline expects a `data/` folder (not included — see below) and generates a `faiss_index/` folder when you build the vector store (also not included; it's a build artifact).

## Setup

```bash
git clone https://github.com/Preshitajais21/multi-agent-research-assistant.git
cd multi-agent-research-assistant

python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
```

### Add your own data

Research PDFs are not included in this repo (they're third-party copyrighted papers). Create a `data/` folder inside `single_agent_pipeline/` and/or `multi_agent_pipeline/` and add your own PDFs there before building the vector store.

### Set your OpenAI API key

Set it as an environment variable — **never commit it to a file in this repo**:

```bash
export OPENAI_API_KEY=your-key-here      # macOS/Linux
setx OPENAI_API_KEY your-key-here        # Windows (restart terminal after)
```

Verify it's set:

```bash
python -c "import os; print('Key found' if os.getenv('OPENAI_API_KEY') else 'No key found')"
```

## Running the Single-Agent Pipeline

```bash
cd single_agent_pipeline
streamlit run app.py
```

## Running the Multi-Agent Pipeline

Build the vector store whenever PDFs are added, removed, or changed:

```bash
cd multi_agent_pipeline
python src/build_vector_store.py
```

Run the full pipeline in the terminal:

```bash
python src/full_pipeline.py
```

This prints the planner output, both candidate answers, the judge's decision, the final selected answer, LLM-as-a-Judge evaluation, RAGAS evaluation, sources used, and token usage.

Or run the Streamlit UI:

```bash
streamlit run app.py
```

## Tech Stack

LangChain, LangGraph, OpenAI API, FAISS, RAGAS, Streamlit, Python.

## Full Write-Up

See [`Multi-Agent_Research_Assistant_Report.docx`](./Multi-Agent_Research_Assistant_Report.docx) for the detailed methodology and comparison results between the two pipelines.
