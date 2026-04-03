import logging
import json
# from ingestionPipeline import IngestionPipeline
from pipeline import RAGPipeline
# from evals.build_golden_dataset import build_golden_dataset
# from evals.ragas import run_ragas

import gradio as gr

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # pipeline = IngestionPipeline()
    # pipeline.run(data_dir="data/full_contract_pdf", data_csv_path="data/master_clauses.csv")

    # pipeline = RAGPipeline()
    # query = "Is there a 'Renewal' clause in 'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.PDF'?"
    # results = pipeline.run(query)
    # answer = pipeline.llm.generate_response(results["rewritten_query"], results["chunks"])
    # print(answer)
    

    # golden = build_golden_dataset()

    # with open("evals/golden_dataset_ragas_ready.json", "w") as f:
    #     json.dump(golden,f,indent=2)

    # logger.info(f"\nSaved {len(golden)} Q&A pairs to evals/golden_dataset.json")

    # run_ragas()

    # logger.info(f"\n RAGAS Evaluation Ended Successfully.")



    # Initialize your pipeline


# Initialize your pipeline


    pipeline = RAGPipeline()

    def process_query(query):
        try:
            result = pipeline.run(query) 
            raw_llm_output = pipeline.llm.generate_response(
                result["rewritten_query"], result["chunks"]
            )
            
            answer_text = raw_llm_output.get("answer", "No analysis available.")
            sources_list = raw_llm_output.get("sources", [])

            # --- Modern Grid-based Source Rendering ---
            resources_html = "<div style='display: flex; flex-direction: column; gap: 24px;'>"
            
            for src in sources_list:
                # Metadata Extraction
                fname = src.get("file_name", "Unknown Contract")
                ctype = src.get("contract_type", "Legal Agreement")
                p1 = src.get("party_1", "N/A")
                p2 = src.get("party_2", "N/A")
                law = src.get("governing_law", "Not Specified")
                notice = src.get("notice_period_to_terminate", "N/A")
                
                # Date Handling
                eff = src.get("effective_date_human_display", "—")
                exp = src.get("expiration_date_human_display", "—")

                resources_html += f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(79, 172, 254, 0.2); border-radius: 16px; padding: 25px; color: white; position: relative; overflow: hidden;">
                    <div style="margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 15px;">
                        <div style="color: #4facfe; font-size: 0.75em; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px;">◈ {ctype}</div>
                        <h2 style="margin: 0; font-size: 1.25em; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{fname}</h2>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        
                        <div style="display: flex; flex-direction: column; gap: 15px;">
                            <div>
                                <div style="color: #6b7280; font-size: 0.7em; text-transform: uppercase; margin-bottom: 4px;">Primary Parties</div>
                                <div style="color: #e5e7eb; font-size: 0.9em; font-weight: 600;">{p1} <span style="color: #4facfe;">vs</span> {p2}</div>
                            </div>
                            <div>
                                <div style="color: #6b7280; font-size: 0.7em; text-transform: uppercase; margin-bottom: 4px;">Governing Law</div>
                                <div style="color: #4facfe; font-size: 0.9em; font-weight: 600;">{law}</div>
                            </div>
                        </div>

                        <div style="display: flex; flex-direction: column; gap: 15px;">
                            <div>
                                <div style="color: #6b7280; font-size: 0.7em; text-transform: uppercase; margin-bottom: 4px;">Notice Period</div>
                                <div style="color: #ffffff; font-size: 0.9em; font-weight: 600;">{notice}</div>
                            </div>
                            <div>
                                <div style="color: #6b7280; font-size: 0.7em; text-transform: uppercase; margin-bottom: 4px;">Contract Timeline</div>
                                <div style="color: #e5e7eb; font-size: 0.85em;">{eff} <span style="color: #4facfe;">→</span> {exp}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, #4facfe, transparent);"></div>
                </div>
                """
            resources_html += "</div>"
            
            return answer_text, resources_html

        except Exception as e:
            return f"### Analysis Error\n{str(e)}", ""

    # --- CSS for "DataCo Blue" Tabs and Modern UI ---
    custom_css = """
    .gradio-container { background-color: #0b1120 !important; font-family: 'Inter', sans-serif !important; }
    #input-box textarea {
    background-color: #0b1120 !important; /* Matches your main background */
    border: 1px solid #1f2937 !important;
    color: white !important;
    border-radius: 12px !important;
    }

    /* Custom Styling for the Tabs */
    .tab-nav { border-bottom: 1px solid #1f2937 !important; justify-content: center !important; margin-bottom: 30px !important; }
    .tab-nav button { 
        color: #9ca3af !important; 
        border: none !important; 
        font-weight: 700 !important; 
        font-size: 1.1em !important; 
        transition: all 0.3s ease !important;
    }
    .tab-nav button.selected { 
        color: #4facfe !important; 
        border-bottom: 3px solid #4facfe !important; 
        background: transparent !important; 
        text-shadow: 0 0 10px rgba(79, 172, 254, 0.5) !important;
    }

    /* Button & Input Styling */
    input[type="text"] { background-color: #111827 !important; border: 1px solid #1f2937 !important; color: white !important; border-radius: 12px !important; }
    button.primary { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important; 
        border: none !important; 
        color: #0b1120 !important; 
        font-weight: 800 !important; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(css=custom_css) as demo:
        with gr.Column():
            gr.HTML("""
                <div style="text-align: center; padding: 60px 0 40px 0;">
                    <p style="color: #4facfe; font-weight: 900; letter-spacing: 4px; font-size: 0.7em; margin-bottom: 10px;">DATACO MOROCCO HUB</p>
                    <h1 style="color: white; font-size: 3.5em; font-weight: 900; margin: 0; line-height: 1;">CONTRACT <span style="color: #4facfe;">AI</span></h1>
                    <p style="color: #6b7280; font-size: 1.1em; margin-top: 15px; font-weight: 500;">Interactive Legal Analytics & Document Intelligence</p>
                </div>
            """)
            
            with gr.Row():
                user_query = gr.Textbox(show_label=False, placeholder="Identify risks in the 2026 Simplicity Agreement...", scale=4, container=False,elem_id="input-box")
                submit_btn = gr.Button("Analyze →", variant="primary", scale=1)

            with gr.Tabs():
                with gr.TabItem("Legal Analysis"):
                    output_answer = gr.Markdown(elem_classes="tab-nav")
                with gr.TabItem("Extracted Sources"):
                    output_resources = gr.HTML(elem_classes="tab-nav")

        submit_btn.click(fn=process_query, inputs=user_query, outputs=[output_answer, output_resources])

    if __name__ == "__main__":
        demo.launch()