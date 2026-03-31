import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# # Covers formats: January 1, 2023 / Jan 1, 2023 / 01/01/2023 / 2023-01-01
# DATE_PATTERN = re.compile(
#     r"""
#     (?:
#         (?:January|February|March|April|May|June|July|August|
#            September|October|November|December|
#            Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)
#         [\s,]+\d{1,2}[\s,]+\d{4}
#     )
#     |
#     (?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})
#     |
#     (?:\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})
#     """,
#     re.VERBOSE | re.IGNORECASE,
# )

# EXPIRY_CONTEXT = re.compile(
#     r"(?:expir|terminat|end\s+date|renewal\s+date|expire[sd]?|expiration).*?"
#     r"(" + DATE_PATTERN.pattern + r")",
#     re.IGNORECASE | re.VERBOSE | re.DOTALL,
# )

# EFFECTIVE_CONTEXT = re.compile(
#     r"(?:effective\s+date|commenc|start\s+date|dated\s+as\s+of|entered\s+into).*?"
#     r"(" + DATE_PATTERN.pattern + r")",
#     re.IGNORECASE | re.VERBOSE | re.DOTALL,
# )

# ## Clause Detection
# CLAUSE_PATTERNS = {
#     "termination": [
#         r"terminate",
#         r"termination",
#         r"terminate\s+this\s+agreement",
#         r"right\s+to\s+terminate"
#     ],

#     "confidentiality": [
#         r"confidential",
#         r"confidentiality",
#         r"non.?disclosure",
#         r"proprietary\s+information"
#     ],

#     "payment": [
#         r"payment",
#         r"fees",
#         r"compensation",
#         r"invoice",
#         r"amount\s+due"
#     ],

#     "liability": [
#         r"liability",
#         r"limitation\s+of\s+liability",
#         r"damages",
#         r"indemnif"
#     ],

#     "governing_law": [
#         r"governed\s+by",
#         r"governing\s+law",
#         r"jurisdiction"
#     ],

#     "intellectual_property": [
#         r"intellectual\s+property",
#         r"copyright",
#         r"patent",
#         r"trademark"
#     ]
# }


# ## parties Detection

# BETWEEN_PATTERN = re.compile(
#     r"between\s+(.+?)\s+and\s+(.+?)(?:\n|\.|\()",
#     re.IGNORECASE | re.DOTALL
# )
# LEGAL_ENTITY_PATTERN = re.compile(
#     r"\b([A-Z][A-Za-z0-9&,\.\-\s]{2,}"
#     r"(?:Inc\.|LLC|Ltd\.|Corporation|Corp\.|Company|S\.A\.|SARL|S\.R\.L\.))\b",
#     re.IGNORECASE
# )
# ALIAS_PATTERN = re.compile(
#     r"([A-Z][A-Za-z0-9\s,\.&\-]+?)\s*\(\s*\"?(Company|Client|Vendor|Supplier|Contractor|Customer)\"?\s*\)",
#     re.IGNORECASE
# )


# def extract_dates(text: str) -> dict:
#     dates = {}

#     match = EFFECTIVE_CONTEXT.search(text[:3000])
#     if match:
#         dates["effective_date"] = match.group(1).strip()

#     match = EXPIRY_CONTEXT.search(text)
#     if match:
#         candidate = match.group(1).strip()
#         # ← reject SEC filing date patterns like "1/16/2020" near "Source:"
#         if "source:" not in text[max(0, match.start()-50):match.start()].lower():
#             dates["expiry_date"] = candidate
#         else:
#             dates["expiry_date"] = ""
#     else:
#         dates["expiry_date"] = ""   # ← explicit empty, not missing key

#     return dates

# def detect_clause(text: str) -> list[str]:
#     found = []
#     for clause_type, patterns in CLAUSE_PATTERNS.items():
#         for pattern in patterns:
#             if re.search(pattern, text, re.IGNORECASE):
#                 found.append(clause_type)
#                 break   # one match per clause type is enough
#     return found if found else ["general"]


# def detect_parties(text):
#     sample = text[:1500]

#     parties = []

#     # 1. Between pattern
#     matches = BETWEEN_PATTERN.findall(sample)
#     for m in matches:
#         parties.extend([p.strip() for p in m if p.strip()])

#     # 2. Alias pattern (Company, Client, etc.)
#     matches = ALIAS_PATTERN.findall(sample)
#     for m in matches:
#         parties.append(m[0].strip())

#     # 3. Legal entities
#     matches = LEGAL_ENTITY_PATTERN.findall(sample)
#     for m in matches:
#         parties.append(m.strip())

#     # Deduplicate
#     seen = set()
#     unique = []
#     for p in parties:
#         if p not in seen and len(p) > 3:
#             seen.add(p)
#             unique.append(p)

#     structured = {
#         "party_1": unique[0] if len(unique) > 0 else "",
#         "party_2": unique[1] if len(unique) > 1 else ""
#     }

#     return structured


## ------- main function --------------
def enrich_metadata_v2(documents):

    logger.info(f"strat enriching metadata for {len(documents)} documents")
    data = pd.read_csv("../data/master_clauses.csv")

    for doc in documents:
        for row in data:
            if doc.metadata.get("source_file") == row["Filename"]:
                #parties:
                parties = re.findall(r'([^;()]+)(?:\s*\(|$)', row["Parties-Answer"])
                parties = [party.strip() for party in parties if party.strip()] 
                party_1 = parties[0]
                party_2 = parties[1]

                # converting dates
                agre_dt_obj = pd.to_datetime(row["Agreement Date-Answer"], format='%m/%d/%y')
                agre__qdrant_date = agre_dt_obj.strftime('%Y-%m-%dT%H:%M:%SZ')

                effec_dt_obj = pd.to_datetime(row["Agreement Date-Answer"], format='%m/%d/%y')
                effec__qdrant_date = effec_dt_obj.strftime('%Y-%m-%dT%H:%M:%SZ')

                exper_dt_obj = pd.to_datetime(row["Agreement Date-Answer"], format='%m/%d/%y')
                exper__qdrant_date = exper_dt_obj.strftime('%Y-%m-%dT%H:%M:%SZ')

                doc.metadata.update({
                    "party_1":party_1,
                    "party_2":party_2,
                    "contract_type":row["Document Name-Answer"],
                    "agreement_date":agre__qdrant_date,
                    "effective_date":effec__qdrant_date,
                    "expiration_date":exper__qdrant_date,
                    "notice_period_to_terminate": row["Notice Period To Terminate Renewal- Answer"],
                    "renewl_term": row["Renewal Term-Answer"],
                    "governing_law": row["Governing Law-Answer"]

                })
    logger.info(f"metadata enriched successfully")
    return documents

# def enrich_metadata(documents):
#     logger.info(f"strat enriching metadata for {len(documents)} documents")
#     for doc in documents:
#         text = doc.page_content
#         text_lower = text.lower()
#         contract_date = extract_dates(text_lower)
#         clause_type = detect_clause(text_lower)
#         parties = detect_parties(text)
#         #enrich metadata
#         doc.metadata.update(contract_date)
#         doc.metadata["Clause_type"] = clause_type
#         doc.metadata.update(parties)
#     logger.info(f"metadata enriched successfully")

#     return documents
    