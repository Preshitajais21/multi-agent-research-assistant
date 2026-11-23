from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, LLMContextPrecisionWithoutReference

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


def split_context(context, max_chunks=6, max_chars_per_chunk=1000):
    if context is None:
        return []

    parts = str(context).split("\n\n")
    chunks = []

    for part in parts:
        part = part.strip()
        if part:
            chunks.append(part[:max_chars_per_chunk])

        if len(chunks) >= max_chunks:
            break

    return chunks


def shorten_answer(answer, max_chars=2500):
    if answer is None:
        return ""
    return str(answer)[:max_chars]


def run_ragas_evaluation(query, answer, context):
    contexts = split_context(context)
    short_answer = shorten_answer(answer)

    data = {
        "question": [query],
        "answer": [short_answer],
        "contexts": [contexts],
    }

    dataset = Dataset.from_dict(data)

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=2048,
        )
    )

    context_precision_no_ref = LLMContextPrecisionWithoutReference(
        llm=ragas_llm
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            context_precision_no_ref,
        ],
        llm=ragas_llm,
        raise_exceptions=False,
        show_progress=True,
    )

    return result