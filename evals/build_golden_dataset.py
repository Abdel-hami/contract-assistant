from datasets import load_dataset
from sympy import re
from pipeline import RAGPipeline
import time
import random
import json
import logging
from qdrant_client import models

logger = logging.getLogger(__name__)

# def load_cuad_qa_flat(json_path:str, n_samples):
#     logger.info("Start converting cuad_qa to flat dataset...")
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     flat_data = []

#     for doc in data["data"]:
#         title = doc.get("title", "")
#         # NORMALIZE: Keep only alphanumeric and spaces to help matching
#         search_friendly_title = "".join([char if char.isalnum() else " " for char in str(title)]).strip()
#         # Remove double spaces
#         search_friendly_title = " ".join(search_friendly_title.split())       
#         for para in doc["paragraphs"]:
#             context = para.get("context", "")

#             for qa in para["qas"]:
#                 if qa.get("is_impossible", False):
#                     continue

#                 answers = qa.get("answers", [])
#                 if not answers:
#                     continue

#                 flat_data.append({
#                     "question": qa["question"],
#                     "ground_truth": answers[0]["text"],
#                     "context": context,
#                     "source_title":search_friendly_title
#                 })

#     logger.info("Converting cuad_qa to flat dataset Done Successfully")

#     random.seed(42)
#     return random.sample(flat_data, min(n_samples, len(flat_data)))


def build_golden_dataset():

    pipeline = RAGPipeline()

    with open("evals/golden_dataset.json", "r") as f:
        cuad = json.load(f)
    # cuad = load_cuad_qa_flat(json_path=json_path, n_samples=n_samples)

    golden = []

    for i, row in enumerate(cuad):
        logger.info(f"Processing {i+1}/{len(cuad)}: {row['question']}")


        try:
            result = pipeline.run(row["question"],top_k=8)
            answer = pipeline.llm.generate_response(
                result["rewritten_query"], result["chunks"]
            )
            time.sleep(5)
            golden.append(
                {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "answer": answer["answer"],
                    "retrieved_contexts": [c["text"] for c in result["chunks"]]
                }
            )

        except Exception as e:
            logger.error(f"skipped: {e}")
            continue

    return golden


