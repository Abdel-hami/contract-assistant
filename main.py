import logging
import json
# from ingestionPipeline import IngestionPipeline
# from pipeline import RAGPipeline
from evals.build_golden_dataset import build_golden_dataset
from evals.ragas import run_ragas

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # pipeline = IngestionPipeline()
    # pipeline.run("data/full_contract_pdf")

    # pipeline = RAGPipeline()
    # query = "Does Bioamber have the right to sublicense the Research License granted in Section 2.5?"
    # results = pipeline.run(query)
    # answer = pipeline.llm.generate_response(results["rewritten_query"], results["chunks"])
    # print(answer)
    

    
    # golden = build_golden_dataset(json_path="data/CUAD_v1.json")

    # with open("evals/golden_dataset.json", "w") as f:
    #     json.dump(golden,f,indent=2)

    # logger.info(f"\nSaved {len(golden)} Q&A pairs to evals/golden_dataset.json")

    run_ragas()

    logger.info(f"\n RAGAS Evaluation Ended Successfully.")