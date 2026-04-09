import requests
import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_gradio():
    API_URI = os.getenv("API_URI", "YOUR_API_ENDPOINT_HERE") 
    
    def process_query(query):
        try:
            response = requests.post(API_URI, json={"query": query, "top_k": 10})
            response.raise_for_status()
            result = response.json()
            
            answer_text = result.get("answer", "No analysis available.")
            sources_list = result.get("sources", [])

            # --- Legal Analysis: Modern Monochrome Glass Card ---
            modern_answer = f"""
            <div class="analysis-card">
                <div class="analysis-glow"></div>
                <div class="analysis-content">
                    {answer_text}
                </div>
            </div>
            """

            # --- Source Intelligence: Latest Dossier Design (Monochrome) ---
            resources_html = "<div class='sources-container'>"
            for src in sources_list:
                file_name = src.get("filename", "Unknown Document")
                ctype = src.get("contract_type", "N/A")
                p1, p2 = src.get("party_1", "N/A"), src.get("party_2", "N/A")
                law = src.get("governing_law", "N/A")
                agr_date = src.get("agreement_date_human_display", "Unknown")
                eff = src.get("effective_date_human_display", "Unknown")
                exp = src.get("expiration_date_human_display", "Unknown")
                notice = src.get("notice_period_to_terminate", "Unknown")

                resources_html += f"""
                <div class="dossier-card">
                    <div class="dossier-accent"></div>
                    <div class="dossier-header">
                        <span class="file-tag">DOCUMENT NAME</span>
                    </div>
                    
                    <h3 class="file-name">{file_name}</h3>
                    
                    <div class="dossier-stats">
                        <div class="stat-box"><label>CONTRACT TYPE</label> <span>{ctype}</span></div>
                        <div class="stat-box"><label>PRIMARY PARTIES</label> <span>{p1} <i class="accent-plus">+</i> {p2}</span></div>
                        <div class="stat-box"><label>GOVERNING LAW</label> {law}</span></div>
                        <div class="stat-box"><label>AGREEMENT DATE</label> <span>{agr_date}</span></div>
                        <div class="stat-box"><label>EFFECTIVE DATE</label> <span>{eff}</span></div>
                        <div class="stat-box"><label>EXPIRATION DATE</label> <span>{exp}</span></div>
                        <div class="stat-box"><label>NOTICE PERIOD TO TERMINATE</label> <span>{notice}</span></div>
                    </div>
                </div>
                """
            resources_html += "</div>"
            
            return modern_answer, resources_html

        except Exception as e:
            return f"<div class='analysis-card' style='border-color: #ff4b4b;'>⚠️ System Error: {str(e)}</div>", ""

    # --- Senior Designer Custom CSS ---
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --emerald: #00f5d4;
        --blue: #4facfe;
        --dark-bg: #050914;
        --glass-bg: rgba(255, 255, 255, 0.03);
    }

    .gradio-container { background-color: var(--dark-bg) !important; font-family: 'Outfit', sans-serif !important; }

    /* Synchronized Search Bar */
    #input-row { align-items: stretch !important; gap: 10px !important; }
    #input-box textarea {
        background: var(--glass-bg) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 18px !important;
        height: 60px !important;
        font-size: 1rem !important;
    }

    .analyze-btn { 
        height: 60px !important;
        background: linear-gradient(135deg, var(--blue), var(--emerald)) !important; 
        border: none !important; 
        color: var(--dark-bg) !important; 
        font-weight: 800 !important; 
        border-radius: 12px !important;
        letter-spacing: 1px;
    }

    /* MONOCHROME TAB SYSTEM (Orange Removed) */
    .tabs .tab-nav { border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important; }
    .tabs .tab-nav button { color: #4b5563 !important; font-weight: 700 !important; border-bottom: 2px solid transparent !important; background: transparent !important; }
    
    /* FIX: Active tab and bar is now WHITE */
    .tabs .tab-nav button.selected { 
        color: white !important; 
        border-bottom: 2px solid white !important; 
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.4) !important;
    }

    /* LEGAL ANALYSIS: Modern Glass Panel */
    .analysis-card {
        background: var(--glass-bg);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        position: relative;
        overflow: hidden;
        margin-top: 10px;
    }
    .analysis-glow {
        position: absolute;
        top: 0; left: 0; width: 3px; height: 100%;
        background: linear-gradient(to bottom, var(--blue), var(--emerald));
        opacity: 0.4;
    }
    .analysis-content { color: white !important; line-height: 1.7; font-size: 1.1rem; }
    .analysis-content strong { color: white !important; font-weight: 700; border-bottom: 1px solid rgba(255,255,255,0.2); }

    /* SOURCE DOSSIER CARDS (Monochrome White Highlights) */
    .sources-container { display: flex; flex-direction: column; gap: 20px; padding-top: 15px; }
    .dossier-card {
        background: var(--glass-bg);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        position: relative;
    }
    .dossier-accent {
        position: absolute;
        top: 25%; left: 0; width: 3px; height: 50%;
        background: var(--blue);
    }
    .file-tag { font-size: 0.75em; font-weight: 800; color: #4b5563; letter-spacing: 2px; }
    .file-name { color: white; margin: 10px 0 25px 0; font-size: 1.2em; font-weight: 700; line-height: 1.4; }

    /* Data Alignment Styles */
    .dossier-stats { display: flex; flex-direction: column; gap: 12px; }
    .stat-box { display: flex; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
    .stat-box label { 
        width: 280px; 
        color: #4b5563; 
        font-family: 'Outfit'; 
        font-size: 0.75rem; 
        font-weight: 800; 
        text-transform: uppercase;
        flex-shrink: 0;
    }

    .highlight-emerald { color: var(--emerald) !important; font-weight: 700; }
    .accent-plus { color: var(--emerald); font-style: normal; margin: 0 4px; }

    footer { display: none !important; }
    """

    with gr.Blocks(css=custom_css, title="DEEP CLAUSE • LEGAL INTELLIGENCE") as demo:
        with gr.Column():
            gr.HTML("""
                <div style="text-align: center; padding: 35px 0;">
                    <h1 style="color: white; font-size: 3em; font-weight: 900; margin: 0; letter-spacing: -2px;">
                        DEEP<span style="color: #4facfe;">CLAUSE</span>
                    </h1>
                    <p style="color: #6b7280; font-size: 1.2em; margin-top: 10px;">Advanced RAG for Legal & Sales Analysis</p>
                </div>
            """)
            
            with gr.Row(elem_id="input-row"):
                user_query = gr.Textbox(
                    show_label=False, 
                    placeholder="Enter query to analyze legal documents...", 
                    scale=4, elem_id="input-box", container=False
                )
                submit_btn = gr.Button("RUN ANALYSIS", scale=1, elem_classes="analyze-btn")

            with gr.Tabs(elem_classes="tabs"):
                with gr.TabItem("Legal Analysis"):
                    output_answer = gr.HTML()
                
                with gr.TabItem("Source Intelligence"):
                    output_resources = gr.HTML()

        submit_btn.click(fn=process_query, inputs=user_query, outputs=[output_answer, output_resources])

    demo.launch()

if __name__ == "__main__":
    run_gradio()