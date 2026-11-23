from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.ragas_evaluation import run_ragas_evaluation


MODEL_NAME = "gpt-5.4-mini"


def load_vector_store(path="faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )


def retrieve_context(query, k=6):
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=k)

    context_parts = []
    sources = []

    for doc in results:
        title = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
        sources.append(title)
        context_parts.append(f"[Source: {title}]\n{doc.page_content}")

    return "\n\n".join(context_parts), sorted(list(set(sources)))


def generate_plan(query):
    client = OpenAI()

    prompt = f"""
You are a Planner Agent for a literature review system.

Break the user's research query into a clear academic literature review outline.

Rules:
- Create 4 to 6 sections
- Keep it concise
- Include research gaps or future directions if relevant
- Do not write the full review
- Only output the plan

User Query:
{query}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text, response.usage


def generate_review_detailed(query, plan, context):
    client = OpenAI()

    prompt = f"""
You are a detailed literature review Writer Agent.

Write a structured literature review using:
1. User query
2. Planner outline
3. Retrieved document context

Rules:
- Use ONLY the retrieved context
- Do NOT invent papers, authors, links, citations, or facts
- If evidence is missing, write "insufficient information"
- Mention source filenames where relevant
- Follow the planner outline closely
- Use clear academic writing
- Provide enough detail for a useful literature review

User Query:
{query}

Planner Outline:
{plan}

Retrieved Context:
{context}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text, response.usage


def generate_review_strict(query, plan, context):
    client = OpenAI()

    prompt = f"""
You are a strict evidence-grounded literature review Writer Agent.

Write a concise literature review using ONLY the retrieved context.

Rules:
- Follow the planner outline structure, but only include sections that are supported by the context.
- Remove weak or unsupported claims
- Prefer accuracy over completeness
- If evidence is missing, write "insufficient information"
- Mention source filenames for each major point in brackets
- Keep the answer concise and academic
- Avoid general background knowledge unless it appears in the retrieved context

User Query:
{query}

Planner Outline:
{plan}

Retrieved Context:
{context}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text, response.usage


def critique_review(query, plan, context, draft):
    client = OpenAI()

    prompt = f"""
You are a Critic Agent for a literature review system.

Evaluate the writer draft using the query, planner outline, and retrieved context.

Check:
1. Factual consistency with the context
2. Completeness compared to the planner outline
3. Clarity and coherence
4. Unsupported or overstated claims
5. Missing source support
6. Places where "insufficient information" should be used

Output format:
1. Strengths
2. Problems Found
3. Missing Points
4. Suggested Improvements
5. Overall Verdict

Do NOT rewrite the full answer.

User Query:
{query}

Planner Outline:
{plan}

Retrieved Context:
{context}

Writer Draft:
{draft}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text, response.usage


def refine_review(query, plan, context, draft, critique):
    client = OpenAI()

    prompt = f"""
You are a Refiner Agent for a literature review system.

Improve the writer draft using the critic feedback.

Your goals:
- Improve grammar, sentence flow, and academic tone
- Remove repetition
- Improve clarity and organization
- Preserve factual consistency with the retrieved context
- Follow the planner outline
- Fix unsupported or overstated claims
- Keep "insufficient information" wherever evidence is lacking
- Do NOT add new facts that are not in the context
- Mention source filenames where relevant
- Prefer fewer well-supported points over many weakly supported points

Output only the final improved literature review.
Do NOT explain what you changed.

User Query:
{query}

Planner Outline:
{plan}

Retrieved Context:
{context}

Writer Draft:
{draft}

Critic Feedback:
{critique}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text, response.usage


def judge_answers(query, plan, context, answer_a, answer_b):
    client = OpenAI()

    prompt = f"""
You are a Judge Agent for a multi-agent literature review system.

You are comparing two final candidate answers generated for the SAME user query.

Candidate A = Response generated using Strategy A
Candidate B = Response generated using Strategy B

Evaluate both candidates using:
1. Factual consistency with retrieved context
2. Completeness with respect to planner outline
3. Coherence and grammar
4. Source grounding
5. Overall usefulness for literature review

Important:
- Do NOT prefer either candidate based on its label.
- Select the answer that best balances factual grounding, completeness, clarity, and usefulness.
- Penalize unsupported claims.
- Penalize answers that are too incomplete or vague.
- Penalize unnecessary repetition or excessive length.

Output in this exact format:

Selected Answer: Candidate A or Candidate B

Scores:
Candidate A: X/5
Candidate B: X/5

Reason:
- ...
- ...
- ...

Final Recommendation:
...

User Query:
{query}

Planner Outline:
{plan}

Retrieved Context:
{context}

Candidate A:
{answer_a}

Candidate B:
{answer_b}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text, response.usage


def choose_selected_answer(judge_output, answer_a, answer_b):
    judge_text = judge_output.lower()

    if "selected answer: candidate a" in judge_text:
        return "Candidate A", answer_a

    if "selected answer: candidate b" in judge_text:
        return "Candidate B", answer_b

    # fallback: choose strict answer if parsing fails
    return "Candidate B", answer_b


def evaluate_answer(query, plan, context, final_answer):
    client = OpenAI()

    prompt = f"""
You are an Evaluation Agent for a RAG-based literature review system.

Evaluate the final selected answer using the user query, planner outline, and retrieved context.

Score each metric from 1 to 5:
1 = poor
5 = excellent

Metrics:
1. Factual Consistency
2. Completeness
3. Coherence
4. Citation Grounding
5. Grammar Quality

Return output exactly like this:

Factual Consistency: X/5
Completeness: X/5
Coherence: X/5
Citation Grounding: X/5
Grammar Quality: X/5
Overall Accuracy Score: X/5

Brief Explanation:
...

User Query:
{query}

Planner Outline:
{plan}

Retrieved Context:
{context}

Final Selected Answer:
{final_answer}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text, response.usage


if __name__ == "__main__":
    while True:
        query = input("\nEnter your research query (or 'exit'): ")

        if query.lower() == "exit":
            break

        print("\nGenerating plan...")
        plan, plan_usage = generate_plan(query)

        print("\nRetrieving context...")
        context, sources = retrieve_context(query, k=6)

        print("\nGenerating Candidate A: detailed writer draft...")
        draft_a, writer_a_usage = generate_review_detailed(query, plan, context)

        print("\nRunning critic for Candidate A...")
        critique_a, critic_a_usage = critique_review(query, plan, context, draft_a)

        print("\nRefining Candidate A...")
        answer_a, refiner_a_usage = refine_review(query, plan, context, draft_a, critique_a)

        print("\nGenerating Candidate B: strict evidence-grounded writer draft...")
        draft_b, writer_b_usage = generate_review_strict(query, plan, context)

        print("\nRunning critic for Candidate B...")
        critique_b, critic_b_usage = critique_review(query, plan, context, draft_b)

        print("\nRefining Candidate B...")
        answer_b, refiner_b_usage = refine_review(query, plan, context, draft_b, critique_b)

        print("\nRunning judge agent...")
        judge_output, judge_usage = judge_answers(query, plan, context, answer_a, answer_b)

        selected_label, selected_answer = choose_selected_answer(judge_output, answer_a, answer_b)

        print("\nEvaluating selected answer with LLM-as-a-Judge...")
        evaluation, eval_usage = evaluate_answer(query, plan, context, selected_answer)

        print("\nRunning RAGAS evaluation on selected answer...")
        ragas_result = run_ragas_evaluation(query, selected_answer, context)

        print("\n===== PLAN =====\n")
        print(plan)

        print("\n===== CANDIDATE A: DETAILED REFINED ANSWER =====\n")
        print(answer_a)

        print("\n===== CANDIDATE B: STRICT REFINED ANSWER =====\n")
        print(answer_b)

        print("\n===== JUDGE AGENT OUTPUT =====\n")
        print(judge_output)

        print("\n===== SELECTED FINAL ANSWER =====")
        print(selected_label)

        print("\n===== FINAL ANSWER RETURNED TO USER =====\n")
        print(selected_answer)

        print("\n===== LLM-AS-A-JUDGE EVALUATION =====\n")
        print(evaluation)

        print("\n===== RAGAS EVALUATION =====")
        print(ragas_result)

        print("\n===== SOURCES USED =====")
        for source in sources:
            print("-", source)

        print("\n===== TOKEN USAGE =====")
        print("Planner:", plan_usage)
        print("Writer A:", writer_a_usage)
        print("Critic A:", critic_a_usage)
        print("Refiner A:", refiner_a_usage)
        print("Writer B:", writer_b_usage)
        print("Critic B:", critic_b_usage)
        print("Refiner B:", refiner_b_usage)
        print("Judge:", judge_usage)
        print("Evaluator:", eval_usage)