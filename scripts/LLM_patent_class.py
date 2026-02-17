import json
import csv
import re
import pandas as pd
from tqdm import tqdm
import time
from flashtext import KeywordProcessor
from openai import OpenAI
import os
from dotenv import load_dotenv
"""
Code is a manual/AI hybrid generated with gemini 3: https://gemini.google.com/share/fbb53ca43596
"""


# --- Initialize Nebius Client (uses openAI system) ---
load_dotenv()
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# --- Load Antibiotic Names ---
antibiotic_file = '../CARD_ontology/antibiotics_CARD_aro.csv'
antibiotics = []
with open(antibiotic_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        antibiotics.append(row[0].lower())


processor = KeywordProcessor(case_sensitive=False)
for name in antibiotics:
    processor.add_keyword(name)


# --- Load Resistance Genes and Terms ---
resistance_gene_file = '../CARD_ontology/aro.tsv'
resistance_genes = pd.read_csv(resistance_gene_file, sep='\t')["Name"]
resistance_terms = ["selection marker", "selectable marker", "marker gene"]


def get_merged_snippets_flashtext(text, processor, window=300):
    """
    Uses FlashText to find keyword locations and merges overlapping windows.
    Returns a list of unique text snippets.
    """
    # extract_keywords with span_info returns: [('keyword', start_idx, end_idx), ...]
    matches = processor.extract_keywords(text, span_info=True)
    
    if not matches:
        return []

    # Calculate raw intervals (start - window, end + window)
    intervals = []
    text_len = len(text)
    
    for _, start, end in matches:
        s_window = max(0, start - window)
        e_window = min(text_len, end + window)
        intervals.append([s_window, e_window])

    # Sort by start position (FlashText usually returns sorted, but safety first)
    intervals.sort(key=lambda x: x[0])

    # Merge overlapping intervals
    merged_snippets = []
    if not intervals:
        return []

    curr_start, curr_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end:
            # Overlap: extend the current window
            curr_end = max(curr_end, next_end)
        else:
            # No overlap: commit current and start new
            merged_snippets.append(text[curr_start:curr_end])
            curr_start, curr_end = next_start, next_end

    # Append the final interval
    merged_snippets.append(text[curr_start:curr_end])
    
    return merged_snippets


# --- llm-classifier function ---
MODEL_NAME="openai/gpt-oss-20b"
PROMPT_TEMPLATE = """
INSTRUCTIONS:
You are a patent analyst. Classify the patent snippet below into one of these categories:
- MARKER: The antibiotic is used as a selection marker or the strain carries resistance.
- AVOIDANCE: The patent describes food-grade markers, marker-free systems, or antibiotic susceptibility of a strain.
- MARKER_AVOIDANCE: Both MARKER and AVOIDANCE aspects are described.
- UNKNOWN: Irrelevant context or general chemical lists.
- BINGO: The antibiotic resistance marker is used during production of food or feed products.

SNIPPET TO ANALYZE:
\"\"\"
{snippet_text}
\"\"\"

RESPONSE FORMAT:
Return ONLY a JSON object: {{"category": "MARKER" | "AVOIDANCE" | "UNKNOWN" | "BINGO", "reason": "one sentence explanation"}}
"""

def categorize_with_llm(text_snippet):
    """Sends the integrated prompt + data in a single request."""
    # Inject snippet into the consolidated template
    full_prompt = PROMPT_TEMPLATE.format(snippet_text=text_snippet)
    
    start_time = time.perf_counter()
    try:
        raw_response = client.chat.completions.with_raw_response.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        headers = raw_response.headers
        remaining_req = headers.get("x-ratelimit-remaining-requests")
        remaining_tok = headers.get("x-ratelimit-remaining-tokens")
        over_limit = headers.get("x-ratelimit-over-limit", "no")

        
        completion = raw_response.parse()
        analysis = json.loads(completion.choices[0].message.content)
        duration = time.perf_counter() - start_time
        analysis.update({
            'input_tokens': completion.usage.prompt_tokens,
            'output_tokens': completion.usage.completion_tokens,
            'duration_sec': round(duration, 3),
            'ratelimit_remaining_req': remaining_req,
            'ratelimit_remaining_tok': remaining_tok,
            'was_over_limit': over_limit
        })
        return analysis
    
    except Exception as e:
        duration = time.perf_counter() - start_time
        return {
            "category": "ERROR",
            "reason": str(e),
            'input_tokens': 0,
            'output_tokens': 0,
            'duration_sec': round(duration, 3),
            'ratelimit_remaining_req': None,
            'ratelimit_remaining_tok': None,
            'was_over_limit': None
        }



path = "/scratch/brbloemen/ARGUMENT/Patent_search/claims_FoodFeedSuppVitEnz_C12N.json"

with open(path, 'r', encoding='utf-8') as f_in:

    snippets = {}
    rows = []

    counter = 0
    for line in tqdm(f_in):
        if not line.strip():
            continue
        
        # store patent information
        patent = json.loads(line)
        lens_id = patent.get('lens_id')
        rows.append({
            'lens_id': lens_id,
            'date_published': patent.get('date_published'),
            'jurisdiction': patent.get('jurisdiction'),
            'doc_number': patent.get('doc_number'),
            'title': patent.get('title', {}).get('text', 'N/A')
        })
        
        # 1. Extract and Process Claims
        claims_data = patent.get('claims', [])
        
        if claims_data != None and len(claims_data) > 0:
            claims_data = claims_data[0]["claims"]
            all_claims_text = " ".join([
                " ".join(claim.get('claim_text', [])) 
                for claim in claims_data
            ])        
            
        else:
            all_claims_text = ""

        # 2. Extract and Process Description
        # Lens descriptions can be a string or a list/dict depending on the version
        desc_data = patent.get('description', {})
        if isinstance(desc_data, dict):
            description_text = desc_data.get('text', "")
        else:
            description_text = str(desc_data)
        
        matches_claims = processor.extract_keywords(all_claims_text)
        matches_desc = processor.extract_keywords(description_text)

        if not matches_claims and not matches_desc:
            continue  # No antibiotic mentions found
        else:
            # extract snippets from patent claims and description
            snippets_claims = get_merged_snippets_flashtext(all_claims_text, processor)
            snippets_desc = get_merged_snippets_flashtext(description_text, processor)
            snippets[lens_id] = {
                'snippets_claims': snippets_claims,
                'snippets_desc': snippets_desc
            }
            # counter += 1
            # if counter >= 50:
            #     break  # test run

    patents = pd.DataFrame(rows)
    patents.to_csv('../results/claimsFoodFeedVitSuppEnz_C12N_patents_metadata.csv', index=False)

test_patent = snippets.popitem()
test_patent_id = test_patent[0]
test_snippet = snippets.popitem()[1]['snippets_desc'][0]
print(test_patent_id)
print(test_snippet)
print(categorize_with_llm(test_snippet)["category"])  # test run

# snippet_classifications = pd.DataFrame(columns=['patent_id', 'snippet_type', 
#                                                 'snippet_text', 'category', 
#                                                 'reason', 'input_tokens', 
#                                                 'output_tokens', 'duration_sec', 
#                                                 'ratelimit_remaining_req', 
#                                                 'ratelimit_remaining_tok', 'was_over_limit'])

# for pat, snip in tqdm(snippets.items()):
#     for claim_snip in snip['snippets_claims']:
#         analysis = categorize_with_llm(claim_snip)
        
#         new_row = pd.DataFrame([{
#             'patent_id': pat,
#             'snippet_type': 'claim',
#             'snippet_text': claim_snip,
#             **analysis
#         }])

#         snippet_classifications = pd.concat([snippet_classifications, new_row], ignore_index=True)
    
#     for desc_snip in snip['snippets_desc']:
#         analysis = categorize_with_llm(desc_snip)
        
#         new_row = pd.DataFrame([{
#             'patent_id': pat,
#             'snippet_type': 'description',
#             'snippet_text': desc_snip,
#             **analysis
#         }])

#         snippet_classifications = pd.concat([snippet_classifications, new_row], ignore_index=True)

# snippet_classifications.to_csv('../results/claimsFoodFeedVitSuppEnz_C12N_snip_class_gptoss20b.csv', index=False)

