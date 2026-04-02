import pandas as pd   
import re
import logging
import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


## Data cleaning functions for metadata enrichmeent

def clean_date_human_display(raw_data:str)->str:
    # 1. Clean the string: Remove brackets and whitespace
    raw_date = str(raw_data).replace("[", "").replace("]", "").strip()
    # 2. Convert to a datetime object
    date_obj = pd.to_datetime(raw_date, format='mixed', errors='coerce')
    # 3. Format it back to your desired string style (MM/DD/YY)
    if pd.notnull(date_obj):
    # %m/%d/%y gives you 07/19/12. 
    # If you want 7/19/12 (no leading zero), use .strftime('%#m/%#d/%y') on Windows
        clean_date_str = date_obj.strftime('%m/%d/%y')
    else:
        clean_date_str = "Unknown" # Or keep it as None/nan
    return clean_date_str

## clean date for qdrant database
def clean_date_iso(raw_data:str):
    # 1. Clean the string: Remove brackets and whitespace
    raw_date = str(raw_data).replace("[", "").replace("]", "").strip()
    # 2. Convert to a datetime object
    date_obj = pd.to_datetime(raw_date, format='mixed', errors='coerce')
    # 3. Format it back to ISO format (YYYY-MM-DD)
    if pd.notnull(date_obj):
        clean_date_str = date_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
    else:
        clean_date_str = "Unknown" 
    return clean_date_str

def clear_nan(raw_value):
    if pd.isna(raw_value):
        return "Unknown"
    else:
        return raw_value
    
def clear_brackets_and_nan(raw_value):
    if pd.isna(raw_value):
        return "Unknown"
    else:
        new_value =  str(raw_value).replace("[", "").replace("]", "").strip()
        if new_value == "":
            return "Unknown"
        return new_value



def enrich_metadata(documents:List[Document], data_csv_path:str):

    logger.info(f"strat enriching metadata for {len(documents)} documents")
    data = pd.read_csv(data_csv_path)

    for doc in documents:
        for _, row in data.iterrows():
            if doc.metadata.get("source_file") == row["Filename"]:
                #parties:
                parties = re.findall(r'([^;()]+)(?:\s*\(|$)', row["Parties-Answer"])
                parties = [party.strip() for party in parties if party.strip()] 
                party_1 = parties[0] if len(parties) > 0 else "Unknown"
                party_2 = parties[1] if len(parties) > 1 else "Unknown"
                doc.metadata.update({
                    "party_1":party_1,
                    "party_2":party_2,
                    "contract_type":row["Document Name-Answer"],
                    "agreement_date":clean_date_iso(row["Agreement Date-Answer"]),
                    "effective_date":clean_date_iso(row["Effective Date-Answer"]),
                    "expiration_date":clean_date_iso(row["Expiration Date-Answer"]),
                    "agreement_date_human_display":clean_date_human_display(row["Agreement Date-Answer"]),
                    "effective_date_human_display":clean_date_human_display(row["Effective Date-Answer"]),
                    "expiration_date_human_display":clean_date_human_display(row["Expiration Date-Answer"]),
                    "notice_period_to_terminate": clear_brackets_and_nan(row["Notice Period To Terminate Renewal- Answer"]),
                    "renewl_term":clear_nan(row["Renewal Term-Answer"]),
                    "governing_law":row["Governing Law-Answer"]

                })
    logger.info(f"metadata enriched successfully.")
    return documents
