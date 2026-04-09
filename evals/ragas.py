from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
import json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

def run_ragas():
    with open("evals/golden_dataset_ragas_ready.json", "r") as f:
        golden = json.load(f)

    dataset = Dataset.from_list(golden)
    model_kwargs = {'device': 'cuda'}  
    encode_kwargs = {'normalize_embeddings': True}
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ChatGroq(model_name="qwen/qwen3-32b", groq_api_key = groq_api_key, temperature=0, n=1, max_retries=10), # n is the number of answers to generate per question and max_retries is the number of times to retry a question if the model fails to generate an answer
        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs=model_kwargs, encode_kwargs =encode_kwargs)
    )

    print(results)

































# i have a problem, for example a question have "of this contract" but my rag app doesn't know wich contract:
#   {
#     "question": "Highlight the parts (if any) of this contract related to \"Non-Disparagement\" that should be reviewed by a lawyer. Details: Is there a requirement on a party not to disparage the counterparty?",
#     "ground_truth": "Franchisee shall not do anything or suffer anything to be done which may adversely affect any rights of Franchisor in and to any Franchisor Property, or any registrations thereof or which, directly or indirectly, may",
#     "answer": "{\n  \"error\": \"This information was not found in the provided contract documents.\"\n}",
#     "context": [
#       "the\u00a0validity\u00a0of\u00a0the\u00a0remaining\u00a0provisions\u00a0of\u00a0this\u00a0Agreement. The\u00a0Parties\u00a0will\u00a0replace\u00a0an\u00a0invalid\u00a0provision\u00a0or\u00a0fill\u00a0any\u00a0gap\u00a0with\u00a0valid\u00a0provisions\u00a0which\nmost\u00a0closely\u00a0approximate\u00a0the\u00a0purpose\u00a0and\u00a0economic\u00a0effect\u00a0of\u00a0the\u00a0invalid\u00a0provision\u00a0or,\u00a0in\u00a0case\u00a0of\u00a0a\u00a0gap,\u00a0the\u00a0Parties\u2019\u00a0presumed\u00a0intentions. In\u00a0the\nevent\u00a0that\u00a0the\u00a0terms\u00a0and\u00a0conditions\u00a0of\u00a0this\u00a0Agreement\u00a0are\u00a0materially\u00a0altered\u00a0as\u00a0a\u00a0result\u00a0of\u00a0the\u00a0preceding\u00a0sentences,\u00a0the\u00a0Parties\u00a0shall\u00a0renegotiate\u00a0the\nterms\u00a0and\u00a0conditions\u00a0of\u00a0this\u00a0Agreement\u00a0in\u00a0order\u00a0to\u00a0resolve\u00a0any\u00a0inequities. Nothing\u00a0in\u00a0this\u00a0Agreement\u00a0shall\u00a0be\u00a0interpreted\u00a0so\u00a0as\u00a0to\u00a0require\u00a0either\nParty\u00a0to\u00a0violate\u00a0any\u00a0applicable\u00a0laws,\u00a0rules\u00a0or\u00a0regulations.",
#       "Each Party shall be responsible for its own costs and expenses in connection with all matters relating to the negotiation and\nperformance of this Agreement, unless otherwise agreed in writing by the Parties. 15.4 Assignment. Neither Newegg nor Allied shall have the right or power to assign or transfer any part of its rights or obligations under\nthis Agreement without the prior consent in writing of the other Party. 15.5 Injunctive Relief. Each Party agrees that money damages for a breach of its obligations under the provisions of this Agreement\nprotecting Confidential Information and those governing Intellectual Property Rights may be an inadequate remedy for the loss suffered by the\nother Party and the other Party shall have the right to obtain injunctive relief from any court of competent jurisdiction in order to prevent the\nbreach, or further breach as the case may be, of any such obligation, without limiting the other Party\u2019s right to pursue any and all remedies provided\nin such event by law or equity. 15.6 Non-Waiver.",
#       "Mutual\u00a0Obligations.",
#       "15.10 Governing Law and Jurisdiction. Without reference to choice or conflict of law principles, this Agreement shall be governed by and\nconstrued in accordance with the laws of the State of California, USA. The Parties unconditionally submit to exclusive jurisdiction of and accept as\nthe exclusive venue for any legal proceeding involving this Agreement the state and federal courts located in the County of Los Angeles,\nCalifornia. Before any Party (the \u201cComplaining Party\u201d) may bring any legal proceeding against the other (the \u201cNon Complaining Party\u201d), the\nComplaining Party shall first make a reasonable and good faith attempt to resolve all disputes privately by notifying and providing to the Non\nComplaining Party of the Complaining Party\u2019s complaints, reasons and supporting evidence for the complaints, and the reasonable steps\nComplaining Party would like the Non Complaining Party to take in order to address the complaints. If for any reason the Non-Complaining Party\ndisagrees with either the complaint or the steps suggested to address the complaints, the Parties shall discuss and work on an amicable solution for\nat least thirty (30) days before the Complaining Party may bring any legal proceeding to resolve the complaints. Any dispute, claim or controversy\narising out of or relating to this Agreement or the breach, termination, enforcement, interpretation, or validity thereof, including the determination of\nthe scope and applicability of this agreement to arbitrate, shall be determined by arbitration in Los Angeles County, California, by an arbitrator of\nJAMS, in accordance with its arbitration rules and procedures then in effect. Judgment on the arbitrator\u2019s award may be entered in any court\nhaving jurisdiction. The prevailing Party in any dispute involving this Agreement shall be entitled to recover from the other Party its costs,\nexpenses, and reasonable attorneys\u2019 fees (including any fees for expert witnesses, paralegals, or other legal service providers). This Section 15.10\nshall not preclude or place any condition on any Party from seeking injunctive relief from a court of appropriate jurisdiction. 15.11 Third Party Rights. This Agreement does not confer any rights or remedies on any third party. 15.12 Counterparts. This Agreement may be executed in any number of counterparts, each of which when executed and delivered shall be\ndeemed to be an original and all of which counterparts taken together shall constitute one and the same instrument. 15.13 Headings. All section headings contained in this Agreement are for convenience or reference only, do not form a part hereof and\nshall not in any way affect the meaning or interpretation of this Agreement.",
#       "Notwithstanding\u00a0anything\u00a0in\u00a0this\u00a0Agreement\u00a0to\u00a0the\u00a0contrary,\u00a0each\u00a0Party\u00a0shall\u00a0have\u00a0the\u00a0right,\u00a0at\u00a0its\u00a0election,\u00a0to\u00a0seek\u00a0injunctive\u00a0or\nother\u00a0equitable\u00a0relief\u00a0in\u00a0any\u00a0court\u00a0of\u00a0competent\u00a0jurisdiction\u00a0to\u00a0enforce\u00a0or\u00a0obtain\u00a0compliance\u00a0with\u00a0any\u00a0provision\u00a0of\nPage\u00a034\u00a0of\u00a060\n[***]\u00a0Certain\u00a0information\u00a0in\u00a0this\u00a0document\u00a0has\u00a0been\u00a0omitted\u00a0and\u00a0filed\u00a0separately\u00a0with\u00a0the\u00a0Securities\u00a0and\u00a0Exchange\u00a0Commission. Confidential\u00a0Treatment\u00a0has\u00a0been\u00a0requested\u00a0with\u00a0respect\u00a0to\u00a0the\u00a0omitted\u00a0portions."
#     ]
#   },
