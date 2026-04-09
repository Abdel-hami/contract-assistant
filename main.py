import logging
import json
# from ingestionPipeline import IngestionPipeline
# from pipeline import RAGPipeline
# from evals.build_golden_dataset import build_golden_dataset
from evals.ragas import run_ragas
from frontend.gradio_interface import run_gradio

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__": 

    # pipeline = IngestionPipeline()
    # pipeline.run(data_dir="data/full_contract_pdf", data_csv_path="data/master_clauses.csv")
    # logger.info(f"\n Ingestion Pipeline Completed Successfully.")

    # pipeline = RAGPipeline()
    # query = "Does 'FOUNDATIONMEDICINE,INC_02_02_2015-EX-10.2-Collaboration Agreement.PDF' contain an 'Audit Rights' clause?"
    # results = pipeline.run(query)
    # answer = pipeline.llm.generate_response(results["rewritten_query"], results["chunks"])
    # print(answer)
    

    # golden = build_golden_dataset()
    # with open("evals/golden_dataset_ragas_ready.json", "w") as f:
    #     json.dump(golden,f,indent=2)
    # logger.info(f"\nSaved {len(golden)} Q&A pairs to evals/golden_dataset.json")


    # run_ragas()
    # logger.info(f"\n RAGAS Evaluation Ended Successfully.")


    run_gradio() 