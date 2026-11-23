import streamlit as st
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.full_pipeline import (
    generate_plan,
    retrieve_context,
    generate_review_detailed,
    generate_review_strict,
    critique_review,
    refine_review,
    judge_answers,
    choose_selected_answer,
    evaluate_answer,
)

from src.ragas_evaluation import run_ragas_evaluation

def extract_judge_scores(text):
    score_a = re.search(r"Candidate A:\s*(\d+)/5", text)
    score_b = re.search(r"Candidate B:\s*(\d+)/5", text)
    selected = re.search(r"Selected Answer:\s*(Candidate [AB])", text)

    return {
        "A": int(score_a.group(1)) if score_a else None,
        "B": int(score_b.group(1)) if score_b else None,
        "selected": selected.group(1) if selected else "Unknown"
    }

st.set_page_config(
    page_title="Multi-Agent Literature Review Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Agent Literature Review Assistant")
st.write("Planner → Retrieval → Writer → Critic → Refiner → Judge → Evaluation")

query = st.text_area(
    "Enter your research query:",
    placeholder="Example: applications and challenges of AI in healthcare",
    height=100
)

run_button = st.button("Run Multi-Agent Pipeline")

if run_button:
    if not query.strip():
        st.warning("Please enter a research query.")
    else:
        with st.spinner("Generating plan..."):
            plan, _ = generate_plan(query)

        with st.spinner("Retrieving relevant context from FAISS..."):
            context, sources = retrieve_context(query, k=6)

        with st.spinner("Generating Candidate A: detailed answer..."):
            draft_a, _ = generate_review_detailed(query, plan, context)
            critique_a, _ = critique_review(query, plan, context, draft_a)
            answer_a, _ = refine_review(query, plan, context, draft_a, critique_a)

        with st.spinner("Generating Candidate B: strict evidence-grounded answer..."):
            draft_b, _ = generate_review_strict(query, plan, context)
            critique_b, _ = critique_review(query, plan, context, draft_b)
            answer_b, _ = refine_review(query, plan, context, draft_b, critique_b)

        with st.spinner("Judge Agent selecting best answer..."):
            judge_output, _ = judge_answers(query, plan, context, answer_a, answer_b)
            selected_label, selected_answer = choose_selected_answer(
                judge_output, answer_a, answer_b
            )

        with st.spinner("Running LLM-as-a-Judge evaluation..."):
            evaluation, _ = evaluate_answer(query, plan, context, selected_answer)

        with st.spinner("Running RAGAS evaluation..."):
            ragas_result = run_ragas_evaluation(query, selected_answer, context)

        st.success("Pipeline completed!")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Final Answer",
            "Plan",
            "Candidates",
            "Judge Output",
            "Evaluation",
            "Sources"
        ])

        with tab1:
            st.subheader("Selected Final Answer")
            st.info(f"Selected by Judge Agent: {selected_label}")
            st.write(selected_answer)

        with tab2:
            st.subheader("Planner Agent Output")
            st.write(plan)

        with tab3:
            st.subheader("Candidate A: Detailed Refined Answer")
            st.write(answer_a)

            st.subheader("Candidate B: Strict Evidence-Grounded Answer")
            st.write(answer_b)

        with tab4:
            st.subheader("Judge Agent Output")
            judge_scores = extract_judge_scores(judge_output)

            st.subheader("Judge Decision")

            st.info(f"Selected Answer: {judge_scores['selected']}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Candidate A", f"{judge_scores['A']}/5")

            with col2:
                st.metric("Candidate B", f"{judge_scores['B']}/5")

            reason = judge_output.split("Reason:")[-1].split("Final Recommendation:")[0].strip()

            st.subheader("Judge Reason")
            st.write(reason)

        with tab5:
            st.subheader("LLM-as-a-Judge Evaluation")
            # extract scores
            def extract_score(text, label):
                match = re.search(rf"{label}:\s*(\d+)/5", text)
                return int(match.group(1)) if match else None

            factual = extract_score(evaluation, "Factual Consistency")
            complete = extract_score(evaluation, "Completeness")
            coherence = extract_score(evaluation, "Coherence")
            citation = extract_score(evaluation, "Citation Grounding")
            grammar = extract_score(evaluation, "Grammar Quality")
            overall = extract_score(evaluation, "Overall Accuracy Score")

            # layout
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Factual", f"{factual}/5")
                st.metric("Completeness", f"{complete}/5")

            with col2:
                st.metric("Coherence", f"{coherence}/5")
                st.metric("Citation", f"{citation}/5")

            with col3:
                st.metric("Grammar", f"{grammar}/5")
                st.metric("Overall", f"{overall}/5")

            # extract explanation
            explanation = evaluation.split("Brief Explanation:")[-1].strip()

            st.subheader("RAGAS Evaluation")
            ragas_df = ragas_result.to_pandas()

            faithfulness = ragas_df["faithfulness"].iloc[0]
            context_precision = ragas_df["llm_context_precision_without_reference"].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Faithfulness", f"{faithfulness:.2f}")

            with col2:
                st.metric("Context Precision", f"{context_precision:.2f}")

            st.subheader("Final Evaluation Summary")
            st.write(explanation)

        with tab6:
            st.subheader("Retrieved Sources")
            for source in sources:
                st.write(f"- {source}")