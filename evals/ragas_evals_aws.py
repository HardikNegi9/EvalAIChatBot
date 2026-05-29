import os
import sys
import types
import json
import asyncio
from dotenv import load_dotenv

# --- BUG FIX FOR RAGAS ---
# Ragas has a hardcoded import for VertexAI that LangChain recently deleted.
# We mock it here so Ragas doesn't crash during initialization.
sys.modules['langchain_community.chat_models.vertexai'] = types.ModuleType('langchain_community.chat_models.vertexai')
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = type('ChatVertexAI', (object,), {})
# -------------------------

# Load env variables BEFORE initializing LangSmith client
load_dotenv(override=True)

from datasets import Dataset
from langsmith import Client
ls_client = Client()
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_aws import ChatBedrockConverse
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.langgraphagenticai.LLMS.awsbedrockllm import AWSBedrockLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from langchain_core.messages import HumanMessage
from src.langgraphagenticai.vectorstore.retriever import Retriever
from langchain_core.tracers.context import collect_runs
import pandas as pd

def run_ragas_evaluation():
    print("Setting up Vector Store...")
    retriever_manager = Retriever()
    asyncio.run(retriever_manager.set_retriever())
    
    print("Setting up Graph...")
    user_controls_input = {"selected_aws_model": "amazon.nova-lite-v1:0"}
    obj_llm_config = AWSBedrockLLM(user_controls_input=user_controls_input)
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

    print("Running AWS Bedrock Graph to collect answers...")
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
                        retrieved_contexts = [doc.page_content for doc in value["documents"]]
            run_id = cb.traced_runs[0].id
        
        questions.append(q)
        answers.append(final_answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(item["expected_answer"])
        run_ids.append(run_id)
        
    print("\nPreparing Ragas Dataset...")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset_hf = Dataset.from_dict(data)
    
    print("Running Ragas Evaluation...")
    print("Running Ragas Evaluation...")
    # Evaluator models are initialized at the top
    # Since Groq hit its rate limit, we will use AWS Bedrock for the Ragas evaluator too!
    eval_llm = ChatBedrockConverse(
        model="amazon.nova-lite-v1:0", 
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        temperature=0
    )
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        result = evaluate(
            dataset_hf,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=eval_llm,
            embeddings=eval_embeddings
        )
        
        print("\n=== RAGAS AWS BEDROCK RESULTS ===")
        print(result)
        
        report_content = f"# Strategy B (AWS Bedrock Baseline) - Ragas Report\n\n"
        report_content += f"**Model:** amazon.nova-lite-v1:0\n\n"
        report_content += f"### Overall Metrics\n"
        
        # In ragas 0.4.x, result is an EvaluationResult object
        if hasattr(result, "items"):
            metrics = result.items()
        elif hasattr(result, "scores"):
            metrics = result.items() if hasattr(result, "items") else [(k, v) for k, v in eval(str(result)).items()]
        else:
            try:
                metrics = dict(eval(str(result))).items()
            except:
                metrics = [("Metrics", "Check console output")]

        for k, v in metrics:
            report_content += f"- **{k}**: {v:.4f}\n"
            
        with open(os.path.join(os.path.dirname(__file__), "strategy_b_aws_report.md"), "w") as f:
            f.write(report_content)
            
        print("\nReport saved to evals/strategy_b_aws_report.md")
        
        # Track this data over time in LangSmith
        print("\nUploading metrics to LangSmith...")
        try:
            df = result.to_pandas()
            for index, row in df.iterrows():
                run_id = run_ids[index]
                for metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    if metric_name in row and not pd.isna(row[metric_name]):
                        ls_client.create_feedback(
                            run_id=run_id,
                            key=metric_name,
                            score=float(row[metric_name])
                        )
            print("LangSmith tracking code executed and metrics uploaded successfully!")
        except Exception as ls_e:
            print(f"Could not upload to LangSmith: {ls_e}")
            
        # CI/CD Threshold Logic
        print("\n=== Checking LLMOps Thresholds ===")
        metrics_dict = dict(metrics)
        fail_pipeline = False
        
        # Check Faithfulness Threshold (Must be >= 0.90)
        faithfulness_score = metrics_dict.get('faithfulness', 0)
        if faithfulness_score < 0.90:
            print(f"[FAILING BUILD] Faithfulness ({faithfulness_score:.4f}) dropped below threshold (0.90)!")
            fail_pipeline = True
        else:
            print(f"[PASSED] Faithfulness ({faithfulness_score:.4f}) >= 0.90")
            
        # Check Context Precision Threshold (Must be >= 0.60)
        precision_score = metrics_dict.get('context_precision', 0)
        if precision_score < 0.60:
            print(f"[FAILING BUILD] Context Precision ({precision_score:.4f}) dropped below threshold (0.60)!")
            fail_pipeline = True
        else:
            print(f"[PASSED] Context Precision ({precision_score:.4f}) >= 0.60")
            
        if fail_pipeline:
            print("\n[ALERT] LLMOps Evaluation Failed! Aborting AWS CodeBuild...")
            sys.exit(1)
        else:
            print("\n[SUCCESS] All LLMOps Evaluations Passed! Ready for deployment.")
            
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ragas_evaluation()
