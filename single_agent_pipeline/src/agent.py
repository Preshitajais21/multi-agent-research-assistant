from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def literature_review_agent(query, context):
    prompt = f"""
You are an academic literature review assistant.

Use ONLY the provided retrieved context.

Tasks:
1. Create literature review outline
2. Write literature review
3. Self critique

Rules:
- Do not hallucinate
- Do not invent citations
- If insufficient information exists, say so

Output format:

## Plan
...

## Literature Review
...

## Self Critique
...

User Query:
{query}

Retrieved Context:
{context}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text