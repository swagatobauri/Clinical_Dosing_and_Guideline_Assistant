import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
from tabulate import tabulate
from dotenv import load_dotenv

# Langchain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# Local imports
from ingestion.chunker import (
    fixed_size_chunker, 
    recursive_chunker, 
    semantic_chunker, 
    hierarchical_chunker
)

load_dotenv()

# Comprehensive Sample Text containing all ground truth answers + ~2000 words padding
SAMPLE_CLINICAL_TEXT = """
Metformin / Renal Dosing Adjustments. The maximum metformin dose for a patient with eGFR 30-45 mL/min/1.73 m2 is 500mg twice daily; do not exceed 1000mg/day in stage 3b CKD (eGFR 30-44).
Apixaban / Atrial Fibrillation Dosing. The recommended apixaban dose for AFib in a patient who is 82 years old, weighs 58 kg, and has a serum creatinine of 1.2 mg/dL is 2.5 mg twice daily, as the patient meets at least two criteria (age >= 80, weight <= 60 kg).
Vancomycin / Pediatric Dosing Guidelines. The initial empiric vancomycin dose for a pediatric patient (age 6) with normal renal function suspected of MRSA bacteremia is 15 to 20 mg/kg/dose every 6 to 8 hours.
Amoxicillin / Acute Otitis Media Pediatric Dosing. The recommended amoxicillin dose for acute otitis media in a child weighing 15 kg is 80 to 90 mg/kg/day divided every 12 hours (approx. 600-675 mg per dose twice daily).
Rosuvastatin / Special Populations. The starting dose of rosuvastatin for a patient of Asian descent is 5 mg once daily due to increased systemic exposure in Asian patients.
Gabapentin / Renal Impairment Adjustments. For a patient with an eGFR < 15 mL/min/1.73 m2 not on dialysis, gabapentin should be dosed at 100 to 300 mg once daily, reduced from the standard dose.
Levothyroxine / Pregnancy Dosing Guidelines. A levothyroxine dose typically should be increased for a patient who is newly pregnant by increasing the pre-pregnancy dose by 20-30%, or administering two additional doses per week.
Fluconazole / Renal Dosing. The fluconazole maintenance dose for oropharyngeal candidiasis in a patient with a creatinine clearance of 30 mL/min is 50% of the normal dose (e.g., 50 to 100 mg daily) after a normal loading dose.
Ketorolac / Contraindications. No, ketorolac is contraindicated in the setting of CABG surgery.
Sildenafil / Contraindications. No, concurrent use of sildenafil and any form of organic nitrates is strictly contraindicated due to the risk of severe hypotension.
Ceftriaxone / Pediatric Contraindications. No, ceftriaxone is contraindicated in hyperbilirubinemic neonates, especially premature ones, due to risk of bilirubin encephalopathy.
Bupropion / Contraindications. No, bupropion is contraindicated in patients with a history of anorexia nervosa or bulimia due to an increased risk of seizures.
Citalopram / Geriatric Dosing and Maximums. The maximum daily dose of citalopram for a 75-year-old patient is 20 mg per day due to the risk of QT prolongation.
Acetaminophen / Maximum Daily Limits. The absolute maximum daily dose of acetaminophen for a healthy adult is 4,000 mg (4 grams) per day to avoid hepatotoxicity.
Lisinopril / Hypertension Dosing Limits. The maximum recommended daily dose of lisinopril for the treatment of hypertension is 80 mg once daily.
Simvastatin / Drug Interactions and Maximum Dosing. The maximum daily dose of simvastatin for a patient who is concurrently taking amlodipine is 20 mg per day to reduce the risk of myopathy.
Warfarin / Drug Interactions. Amiodarone inhibits warfarin metabolism, increasing INR and bleeding risk. The warfarin dose should generally be reduced by 30-50% upon amiodarone initiation.
Clopidogrel / CYP2C19 Interactions. Omeprazole inhibits CYP2C19, which is needed to convert clopidogrel to its active metabolite, thereby significantly reducing clopidogrel's antiplatelet effect.
Digoxin / Therapeutic Drug Monitoring. The target therapeutic serum concentration range for digoxin when treating heart failure is 0.5 to 0.9 ng/mL.
Lithium / Monitoring Parameters. Every 3 to 6 months, or whenever there is a dose change, signs of toxicity, or starting a medication that interacts with lithium.

""" + ("""Metformin is an oral antidiabetic medication used to treat type 2 diabetes. It is a biguanide that works by decreasing hepatic glucose production, decreasing intestinal absorption of glucose, and improving insulin sensitivity by increasing peripheral glucose uptake and utilization. It is typically the first-line medication for the treatment of type 2 diabetes, particularly in people who are overweight. It is also used in the treatment of polycystic ovary syndrome. Metformin is not associated with weight gain and is taken by mouth. Common side effects include diarrhea, nausea, and abdominal pain. It has a low risk of causing low blood sugar. High blood lactic acid level is a concern if the drug is prescribed inappropriately and in overly large doses. It should not be used in those with significant liver disease or kidney problems. Metformin was discovered in 1922. French physician Jean Sterne began study in humans in the 1950s. It was introduced as a medication in France in 1957 and the United States in 1995. It is on the World Health Organization's List of Essential Medicines. """ * 20)

def load_queries(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation_for_strategy(strategy_name: str, chunker_func, queries: List[Dict]):
    print(f"\n--- Running Evaluation for Strategy: {strategy_name} ---")
    
    # 1. Chunking
    sections = [{"text": SAMPLE_CLINICAL_TEXT, "source": "Clinical_Sample.txt", "section": "All", "page": 1}]
    
    try:
        chunks = chunker_func(sections)
        print(f"Generated {len(chunks)} chunks using {strategy_name}.")
    except Exception as e:
        print(f"Chunking failed for {strategy_name}: {e}")
        return {}
    
    if not chunks:
        print("No chunks generated.")
        return {}

    # 2. Build FAISS index
    print("Building FAISS index...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"FAISS indexing failed (missing API key?): {e}")
        return {}
    
    # 3. Setup LLM
    print("Setting up LLM...")
    try:
        llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        llm = ChatGroq(model=llm_model, temperature=0)
    except Exception as e:
        print(f"LLM setup failed (missing GROQ_API_KEY?): {e}")
        return {}
    
    qa_prompt = PromptTemplate.from_template(
        "Use the following pieces of context to answer the clinical question. If you don't know the answer, just say that you don't know.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    # 4. Generate answers
    print("Generating answers for queries...")
    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for i, q in enumerate(queries):
        question = q["query"]
        ground_truth = q["ground_truth_answer"]
        
        # Retrieve chunks
        try:
            retrieved_docs = retriever.invoke(question)
            contexts = [d.page_content for d in retrieved_docs]
            
            # Generate Answer
            context_str = "\n\n".join(contexts)
            prompt_val = qa_prompt.format(context=context_str, question=question)
            response = llm.invoke(prompt_val)
            answer = response.content
            
            dataset_dict["question"].append(question)
            dataset_dict["answer"].append(answer)
            dataset_dict["contexts"].append(contexts)
            dataset_dict["ground_truth"].append(ground_truth)
        except Exception as e:
            print(f"Error processing query {i+1}: {e}")
            
    if not dataset_dict["question"]:
        print("No answers generated.")
        return {}
        
    dataset = Dataset.from_dict(dataset_dict)
    
    # 5. RAGAS Evaluation
    print("Running RAGAS metrics evaluation (requires OPENAI_API_KEY)...")
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
    
    avg_scores = {}
    try:
        results = evaluate(dataset, metrics=metrics)
        avg_scores = {
            "context_precision": results.get("context_precision", 0.0),
            "context_recall": results.get("context_recall", 0.0),
            "faithfulness": results.get("faithfulness", 0.0),
            "answer_relevancy": results.get("answer_relevancy", 0.0),
        }
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        avg_scores = {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        }
    
    # Save results
    os.makedirs("eval/results", exist_ok=True)
    with open(f"eval/results/{strategy_name}_scores.json", "w") as f:
        json.dump(avg_scores, f, indent=2)
        
    return avg_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies with RAGAS")
    parser.add_argument("--strategy", type=str, default="all", choices=["fixed", "recursive", "semantic", "hierarchical", "all"], help="Strategy to evaluate")
    args = parser.parse_args()
    
    queries = load_queries("eval/labeled_queries.json")
    
    strategies = {
        "fixed": lambda s: fixed_size_chunker(s, chunk_size=512, overlap=50),
        "recursive": recursive_chunker,
        "semantic": semantic_chunker,
        "hierarchical": hierarchical_chunker
    }
    
    targets = list(strategies.keys()) if args.strategy == "all" else [args.strategy]
    
    final_results = []
    
    for st in targets:
        func = strategies[st]
        scores = run_evaluation_for_strategy(st, func, queries)
        
        if not scores:
            scores = {"context_precision": 0, "context_recall": 0, "faithfulness": 0, "answer_relevancy": 0}
            
        avg_val = sum(scores.values()) / 4.0 if sum(scores.values()) > 0 else 0.0
        
        final_results.append([
            st,
            round(scores.get("context_precision", 0), 2),
            round(scores.get("context_recall", 0), 2),
            round(scores.get("faithfulness", 0), 2),
            round(scores.get("answer_relevancy", 0), 2),
            round(avg_val, 2)
        ])
        
    print("\n\n==========================================")
    print("Chunking Strategy Benchmark Results")
    print("==========================================")
    headers = ["Strategy", "C.Precision", "C.Recall", "Faithfulness", "Ans.Relevancy", "Avg"]
    print(tabulate(final_results, headers=headers, tablefmt="pipe"))

if __name__ == "__main__":
    main()
