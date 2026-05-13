import pandas as pd
import json
from tqdm import tqdm
import time
import csv
import logging
from openai import OpenAI
import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

# --- Provider selection ---
parser = argparse.ArgumentParser(description="Classify patents with an LLM.")
parser.add_argument(
    "--provider",
    choices=["nebius", "google"],
    default="nebius",
    help="LLM provider: 'nebius' (default) or 'google' (Google AI Studio).",
)
parser.add_argument(
    "--model",
    default=None,
    help="Override the default model for the chosen provider.",
)
parser.add_argument(
    "--input",
    default="/scratch/brbloemen/ARGUMENT/Patent_BLAST/lens_patents/aada1-blast-patents.jsonl",
    help="Path to the input JSONL file.",
)
parser.add_argument(
    "--max-retries",
    type=int,
    default=15,
    help="Maximum retries per patent before recording an error (default: 15).",
)
args = parser.parse_args()

if args.provider == "google":
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    MODEL_NAME = args.model or "gemma-4-31b-it"
    OUTPUT_TAG = MODEL_NAME.replace("/", "_")
else:
    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )
    MODEL_NAME = args.model or "google/gemma-3-27b-it"
    OUTPUT_TAG = MODEL_NAME.replace("/", "_")

INPUT_BASENAME = os.path.splitext(os.path.basename(args.input))[0]
os.makedirs('../results', exist_ok=True)
OUTPUT_PATH = f'../results/{INPUT_BASENAME}_claims_classifications_{OUTPUT_TAG}.tsv'
LOG_PATH = f'../results/{INPUT_BASENAME}_classification_{OUTPUT_TAG}.log'

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# --- llm-classifier function ---
PROMPT_TEMPLATE = """
INSTRUCTIONS:
You are a patent analyst. From the following patent abstract and claims, identify:
(1) the end product (as specifically as possible),
(2) type of production organism (if mentioned). This can be plant, bacteria, yeast, animal, human. If bacterium or yeast, specify the species or genus if and only if a single species or genus is mentioned. Otherwise default to the default "bacterium" or "yeast". If no production organism is mentioned, return "unknown".
(3) Category of end product, one of: "Amino acid", "oligosaccharide", "vitamin", "food colour/flavour", "enzyme", "peptide", "vector", "other"
(4) Sector: whether it falls into one of: "food/feed" (including food industry enzymes, amino acids), "medicinal", "diagnostic", "molecular biology", "chemistry". If it doesn't clearly fall into one of these, classify as "other". When multiple categories apply and food/feed is one of them, choose food/feed.
(5) Provide one-sentence justification for the classification.

SNIPPET TO ANALYZE:
\"\"\"
{snippet_text}
\"\"\"

RESPONSE FORMAT:
Return ONLY a JSON object: {{"End_Product": "end product", "Organism": "organism", "Product_category": "product category" , "Sector": "identified sector", "reason": "one sentence explanation"}}
"""

RETRY_STATUS = {500, 502, 503, 504, 429}
MAX_RETRIES = args.max_retries

# Schema mirrors the JSON object specified in PROMPT_TEMPLATE.
PRODUCT_CATEGORIES = ["Amino acid", "oligosaccharide", "vitamin", "food colour/flavour", "enzyme", "peptide", "vector", "other"]
SECTORS = ["food/feed", "medicinal", "diagnostic", "molecular biology", "chemistry", "other"]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "End_Product": {"type": "string"},
        "Organism": {"type": "string"},
        "Product_category": {"type": "string", "enum": PRODUCT_CATEGORIES},
        "Sector": {"type": "string", "enum": SECTORS},
        "reason": {"type": "string"},
    },
    "required": ["End_Product", "Organism", "Product_category", "Sector", "reason"],
    "propertyOrdering": ["End_Product", "Organism", "Product_category", "Sector", "reason"],
}


def _call_google(full_prompt):
    """One Google GenAI call. Returns (analysis_dict_without_meta, usage_dict)."""
    config = genai_types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        thinking_config=genai_types.ThinkingConfig(thinking_level="MINIMAL"),
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
        config=config,
    )
    analysis = json.loads(response.text)
    usage = response.usage_metadata
    return analysis, {
        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "ratelimit_remaining_req": None,
        "ratelimit_remaining_tok": None,
        "was_over_limit": None,
    }


def _call_openai_compat(full_prompt):
    """One OpenAI-compatible (Nebius) call. Returns (analysis_dict_without_meta, usage_dict)."""
    raw_response = client.chat.completions.with_raw_response.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": full_prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    headers = raw_response.headers
    completion = raw_response.parse()
    analysis = json.loads(completion.choices[0].message.content)
    return analysis, {
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
        "ratelimit_remaining_req": headers.get("x-ratelimit-remaining-requests"),
        "ratelimit_remaining_tok": headers.get("x-ratelimit-remaining-tokens"),
        "was_over_limit": headers.get("x-ratelimit-over-limit", "no"),
    }


def categorize_with_llm(text_snippet):
    """Sends the integrated prompt + data in a single request, with retry on transient errors."""
    full_prompt = PROMPT_TEMPLATE.format(snippet_text=text_snippet)
    call_fn = _call_google if args.provider == "google" else _call_openai_compat

    start_time = time.perf_counter()
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            analysis, usage = call_fn(full_prompt)
            duration = time.perf_counter() - start_time
            if attempt > 0:
                log.info(f"Succeeded after {attempt} retries ({duration:.1f}s)")
            analysis.update({
                **usage,
                'duration_sec': round(duration, 3),
                'retries': attempt,
            })
            return analysis

        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status not in RETRY_STATUS and not isinstance(e, json.JSONDecodeError):
                log.warning(f"Non-retryable error (status={status}): {e}")
                break
            if attempt == MAX_RETRIES - 1:
                log.error(f"Exhausted {MAX_RETRIES} retries. Last error: {e}")
                break

            if status == 429:
                # Google embeds retryDelay in the error body; OpenAI-compatible APIs do not
                import re as _re
                m = _re.search(r"'retryDelay':\s*'(\d+)s'", str(e)) if args.provider == "google" else None
                retry_delay = int(m.group(1)) if m else 0
                backoff = max(retry_delay, 2 ** attempt + 0.5 * attempt)
                # After 10 consecutive 429s on this patent, add 1 hour on top
                if attempt >= 9:
                    backoff += 3600
                    log.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} — 10+ consecutive quota errors, adding 1h. Retrying in {backoff:.0f}s")
                else:
                    log.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed (status=429) — retrying in {backoff:.0f}s (API suggested {retry_delay}s)")
            else:
                backoff = 2 ** attempt + 0.5 * attempt
                log.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed (status={status}): {e} — retrying in {backoff:.1f}s")
            time.sleep(backoff)

    duration = time.perf_counter() - start_time
    return {
        "category": "ERROR",
        "reason": str(last_err),
        'input_tokens': 0,
        'output_tokens': 0,
        'duration_sec': round(duration, 3),
        'ratelimit_remaining_req': None,
        'ratelimit_remaining_tok': None,
        'was_over_limit': None,
        'retries': MAX_RETRIES,
    }


OUTPUT_COLUMNS = [
    'patent_id', 'type', 'text', 'End_Product', 'Organism', 'Product_category',
    'Sector', 'reason', 'input_tokens', 'output_tokens', 'duration_sec',
    'ratelimit_remaining_req', 'ratelimit_remaining_tok', 'was_over_limit', 'retries',
]

# --- Load checkpoint: patents already classified in a previous run ---
already_done = set()
output_exists = os.path.exists(OUTPUT_PATH)
if output_exists:
    try:
        checkpoint = pd.read_csv(OUTPUT_PATH, sep='\t', usecols=['patent_id'])
        already_done = set(checkpoint['patent_id'].astype(str))
        log.info(f"Resuming: {len(already_done)} patents already classified, skipping them.")
    except Exception as e:
        log.warning(f"Could not read checkpoint file: {e} — starting fresh.")
        output_exists = False


with open(args.input, 'r', encoding='utf-8-sig') as f_in:

    patents = {}
    rows = []

    for line in tqdm(f_in):
        if not line.strip():
            continue

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
        if claims_data is not None and len(claims_data) > 0:
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

        if not all_claims_text and not description_text:
            continue
        else:
            patents[lens_id] = {
                'claims': all_claims_text,
                'desc': description_text
            }
        # counter += 1
        # if counter >= 50:
        #     break  # test run

    patent_metadata = pd.DataFrame(rows)
    patent_metadata.to_csv('../results/Patent_Metadata.tsv', sep="\t", index=False)

to_classify = {k: v for k, v in patents.items() if str(k) not in already_done}
log.info(f"Patents to classify this run: {len(to_classify)} (total in file: {len(patents)})")

consecutive_errors = 0
MAX_CONSECUTIVE_ERRORS = 15

# Open output file in append mode; write header only if starting fresh
with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=OUTPUT_COLUMNS, delimiter='\t', extrasaction='ignore')
    if not output_exists:
        writer.writeheader()

    total_input_tokens = 0
    total_output_tokens = 0

    for pat, text in tqdm(to_classify.items()):
        analysis = categorize_with_llm(text['claims'])

        row = {
            'patent_id': pat,
            'type': 'claim',
            'text': text['claims'].replace('\n', ' '),
            **analysis
        }
        writer.writerow(row)
        f_out.flush()

        if analysis.get('category') == 'ERROR':
            consecutive_errors += 1
            log.error(f"Patent {pat} failed: {analysis['reason']} ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} consecutive errors)")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error(f"Stopping: {MAX_CONSECUTIVE_ERRORS} consecutive errors reached. Resume with the same command.")
                break
        else:
            consecutive_errors = 0
            total_input_tokens += analysis.get('input_tokens', 0)
            total_output_tokens += analysis.get('output_tokens', 0)

log.info(f"Run complete. Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")
