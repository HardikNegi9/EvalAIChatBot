import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from datasets import Dataset
from langsmith import Client
ls_client = Client()
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from langchain_core.messages import HumanMessage
from src.langgraphagenticai.vectorstore.retriever import Retriever

load_dotenv(override=True)

def run_ragas_evaluation():
    print("Setting up Vector Store...")
    retriever_manager = Retriever()
    asyncio.run(retriever_manager.set_retriever())
    
    print("Setting up Graph...")
    user_controls_input = {"selected_groq_model": "llama-3.3-70b-versatile"}
    obj_llm_config = GroqLLM(user_controls_input=user_controls_input)
    model = obj_llm_config.get_llm_model()
    
    graph_builder = GraphBuilder(model=model, user_controls_input=user_controls_input)
    graph = graph_builder.setup_graph()

    dataset_path = os.path.join(os.path.dirname(__file__), "qa_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    run_ids = []
    
    print("Running Groq Graph to collect answers...")
    from langchain_core.tracers.context import collect_runs
    
    for idx, item in enumerate(dataset):
        q = item["question"]
        print(f"\nProcessing Q{idx+1}: {q}")
        
        inputs = {"question": q, "generation": "", "documents": [], "generated": 0}
        final_answer = ""
        retrieved_contexts = []
        
        with collect_runs() as cb:
            for output in graph.stream(inputs):
                for key, value in output.items():
                    if key == "generate" and "generation" in value:
                        final_answer = value["generation"]
                    if "documents" in value and not retrieved_contexts:
                        # extract text from docs
                        retrieved_contexts = [doc.page_content for doc in value["documents"]]
            
            # The root run ID for this LangGraph execution
            if cb.traced_runs:
                run_ids.append(str(cb.traced_runs[0].id))
            else:
                run_ids.append(None)
        
        questions.append(q)
        answers.append(final_answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(item["expected_answer"])
        
    print("\nPreparing Ragas Dataset...")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset_hf = Dataset.from_dict(data)
    
    print("Running Ragas Evaluation (Faithfulness, Answer Relevancy, Context Precision, Context Recall)...")
    print("Running Ragas Evaluation (Faithfulness, Answer Relevancy, Context Precision, Context Recall)...")
    eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        result = evaluate(
            dataset_hf,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=eval_llm,
            embeddings=eval_embeddings
        )
        
        print("\n=== RAGAS GROQ RESULTS ===")
        print(result)
        
        # Save to file
        report_content = f"# Strategy A (Groq Baseline) - Ragas Report\n\n"
        report_content += f"**Model:** llama-3.3-70b-versatile\n\n"
        report_content += f"### Overall Metrics\n"
        # In ragas 0.4.x, result is an EvaluationResult object
        if hasattr(result, "items"):
            metrics = result.items()
        elif hasattr(result, "scores"):
            # if scores is a list of dicts, we average them, but usually str(result) gives the dict.
            # Ragas prints it nicely, let's just parse it or iterate over its fields.
            metrics = result.items() if hasattr(result, "items") else [(k, v) for k, v in eval(str(result)).items()]
        else:
            try:
                metrics = dict(eval(str(result))).items()
            except:
                metrics = [("Metrics", "Check console output")]

        for k, v in metrics:
            report_content += f"- **{k}**: {v:.4f}\n"
            
        with open(os.path.join(os.path.dirname(__file__), "strategy_a_groq_report.md"), "w") as f:
            f.write(report_content)
            
        print("\nReport saved to evals/strategy_a_groq_report.md")
        
        # Track this data over time in LangSmith
        print("\nUploading metrics to LangSmith...")
        try:
            df = result.to_pandas()
            for index, row in df.iterrows():
                r_id = run_ids[index]
                if r_id:
                    # Ragas outputs NaN for failed evaluations, so we check using pandas
                    import pandas as pd
                    if not pd.isna(row.get("faithfulness")):
                        ls_client.create_feedback(run_id=r_id, key="faithfulness", score=float(row["faithfulness"]))
                    if not pd.isna(row.get("answer_relevancy")):
                        ls_client.create_feedback(run_id=r_id, key="answer_relevancy", score=float(row["answer_relevancy"]))
                    if not pd.isna(row.get("context_precision")):
                        ls_client.create_feedback(run_id=r_id, key="context_precision", score=float(row["context_precision"]))
                    if not pd.isna(row.get("context_recall")):
                        ls_client.create_feedback(run_id=r_id, key="context_recall", score=float(row["context_recall"]))
                        
            print("Successfully uploaded Ragas metrics as feedback to LangSmith!")
            print("Log into your LangSmith dashboard, click your project, and check the 'Feedback' columns.")
        except Exception as ls_e:
            print(f"Could not upload to LangSmith: {ls_e}")
            
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")

if __name__ == "__main__":
    run_ragas_evaluation()
