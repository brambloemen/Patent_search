import json
import csv
import re
import pandas as pd
from tqdm import tqdm
import time
import logging
import argparse
import os
from flashtext import KeywordProcessor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Argument parsing ---
parser = argparse.ArgumentParser(
    description="Detect antibiotic mentions in Lens.org patent JSONL files and classify each snippet with an LLM."
)
parser.add_argument(
    "--input",
    required=True,
    help="Path to the input JSONL file (one Lens.org patent record per line).",
)
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
    "--antibiotic-list",
    default="../CARD_ontology/antibiotics_list.txt",
    help="Path to the antibiotic name list (one name per line).",
)
parser.add_argument(
    "--window",
    type=int,
    default=300,
    help="Context window in characters around each antibiotic mention (default: 300).",
)
parser.add_argument(
    "--max-retries",
    type=int,
    default=15,
    help="Maximum retries per snippet before recording an error (default: 15).",
)
args = parser.parse_args()

# --- Provider / client setup ---
if args.provider == "google":
    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    MODEL_NAME = args.model or "gemma-4-31b-it"
else:
    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )
    MODEL_NAME = args.model or "openai/gpt-oss-20b"

OUTPUT_TAG = MODEL_NAME.replace("/", "_")
INPUT_BASENAME = os.path.splitext(os.path.basename(args.input))[0]
OUTPUT_PATH = f"../results/{INPUT_BASENAME}_snip_class_{OUTPUT_TAG}.tsv"
LOG_PATH = f"../results/{INPUT_BASENAME}_snip_class_{OUTPUT_TAG}.log"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# --- Load antibiotic keyword list ---
antibiotics = []
with open(args.antibiotic_list, "r") as f:
    for line in f:
        antibiotics.append(line.rstrip().lower())

processor = KeywordProcessor(case_sensitive=False)
for name in antibiotics:
    processor.add_keyword(name)

log.info(f"Loaded {len(antibiotics)} antibiotic keywords from {args.antibiotic_list}")


def get_merged_snippets(text, processor, window=300):
    """Find antibiotic keywords, build context windows, merge overlapping ones."""
    matches = processor.extract_keywords(text, span_info=True)
    if not matches:
        return []

    text_len = len(text)
    intervals = sorted(
        [max(0, s - window), min(text_len, e + window)]
        for _, s, e in matches
    )

    merged = []
    curr_s, curr_e = intervals[0]
    for next_s, next_e in intervals[1:]:
        if next_s <= curr_e:
            curr_e = max(curr_e, next_e)
        else:
            merged.append(text[curr_s:curr_e])
            curr_s, curr_e = next_s, next_e
    merged.append(text[curr_s:curr_e])
    return merged


# --- LLM prompt ---
PROMPT_TEMPLATE = """You are a patent analyst. Classify the antibiotic-related patent snippet below.

CATEGORIES:
- BINGO: Antibiotic resistance marker used in a food/feed production strain
- MARKER: Antibiotic used as bacterial selection marker
- AVOIDANCE: Non-antibiotic marker, marker-free system, marker removal, or antibiotic susceptibility
- MARKER_AVOIDANCE: Both MARKER and AVOIDANCE aspects present
- EUKARYOTIC: Antibiotic used in eukaryotic context
- UNKNOWN: Irrelevant context or general chemical list

SNIPPET:
\"\"\"{snippet_text}\"\"\"

Return ONLY JSON: {{"category": "BINGO"|"MARKER"|"AVOIDANCE"|"MARKER_AVOIDANCE"|"EUKARYOTIC"|"UNKNOWN", "reason": "one sentence"}}"""

RETRY_STATUS = {500, 502, 503, 504, 429}
MAX_RETRIES = args.max_retries

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["BINGO", "MARKER", "AVOIDANCE", "MARKER_AVOIDANCE", "EUKARYOTIC", "UNKNOWN"],
        },
        "reason": {"type": "string"},
    },
    "required": ["category", "reason"],
}


def _call_google(full_prompt):
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
    """Classify a single snippet, with exponential-backoff retry on transient errors."""
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
            analysis.update({**usage, "duration_sec": round(duration, 3), "retries": attempt})
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
                m = re.search(r"'retryDelay':\s*'(\d+)s'", str(e)) if args.provider == "google" else None
                retry_delay = int(m.group(1)) if m else 0
                backoff = max(retry_delay, 2 ** attempt + 0.5 * attempt)
                if attempt >= 9:
                    backoff += 3600
                    log.warning(f"Attempt {attempt+1}/{MAX_RETRIES} — 10+ consecutive quota errors, adding 1h. Retrying in {backoff:.0f}s")
                else:
                    log.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed (429) — retrying in {backoff:.0f}s")
            else:
                backoff = 2 ** attempt + 0.5 * attempt
                log.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed (status={status}): {e} — retrying in {backoff:.1f}s")
            time.sleep(backoff)

    duration = time.perf_counter() - start_time
    return {
        "category": "ERROR",
        "reason": str(last_err),
        "input_tokens": 0,
        "output_tokens": 0,
        "duration_sec": round(duration, 3),
        "ratelimit_remaining_req": None,
        "ratelimit_remaining_tok": None,
        "was_over_limit": None,
        "retries": MAX_RETRIES,
    }


OUTPUT_COLUMNS = [
    "patent_id", "snippet_type", "snippet_text", "category", "reason",
    "input_tokens", "output_tokens", "duration_sec",
    "ratelimit_remaining_req", "ratelimit_remaining_tok", "was_over_limit", "retries",
]

# --- Checkpoint: skip snippets already classified in a previous run ---
already_done = set()
output_exists = os.path.exists(OUTPUT_PATH)
if output_exists:
    try:
        checkpoint = pd.read_csv(OUTPUT_PATH, sep="\t", usecols=["patent_id", "snippet_type", "snippet_text"])
        already_done = set(zip(
            checkpoint["patent_id"].astype(str),
            checkpoint["snippet_type"],
            checkpoint["snippet_text"],
        ))
        log.info(f"Resuming: {len(already_done)} snippets already classified, skipping them.")
    except Exception as e:
        log.warning(f"Could not read checkpoint file: {e} — starting fresh.")
        output_exists = False

# --- Read and keyword-filter patents ---
log.info(f"Reading patents from {args.input}")
snippets = {}

with open(args.input, "r", encoding="utf-8-sig") as f_in:
    for line in tqdm(f_in, desc="Scanning patents"):
        if not line.strip():
            continue
        patent = json.loads(line)
        lens_id = patent.get("lens_id")

        claims_data = patent.get("claims", [])
        if claims_data:
            all_claims_text = " ".join(
                " ".join(claim.get("claim_text", []))
                for claim in claims_data[0]["claims"]
            )
        else:
            all_claims_text = ""

        desc_data = patent.get("description", {})
        description_text = desc_data.get("text", "") if isinstance(desc_data, dict) else str(desc_data)

        snips_claims = get_merged_snippets(all_claims_text, processor, args.window)
        snips_desc = get_merged_snippets(description_text, processor, args.window)

        if snips_claims or snips_desc:
            snippets[lens_id] = {"claims": snips_claims, "desc": snips_desc}

total_snippets = sum(len(v["claims"]) + len(v["desc"]) for v in snippets.values())
total_chars = sum(
    len(s) for v in snippets.values() for s in v["claims"] + v["desc"]
)
log.info(f"Patents with antibiotic mentions: {len(snippets)}")
log.info(f"Total snippets to classify: {total_snippets}")
log.info(
    f"Estimated tokens: "
    f"{(total_snippets * len(PROMPT_TEMPLATE) + total_chars) / 4 / 1e6:.2f}M "
    f"(~4 chars/token)"
)

# --- Classify ---
consecutive_errors = 0
MAX_CONSECUTIVE_ERRORS = 15

with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=OUTPUT_COLUMNS, delimiter="\t", extrasaction="ignore")
    if not output_exists:
        writer.writeheader()

    total_input_tokens = 0
    total_output_tokens = 0

    for pat, snip in tqdm(snippets.items(), desc="Classifying patents"):
        for snip_type, snip_list in [("claim", snip["claims"]), ("description", snip["desc"])]:
            for text in snip_list:
                if (str(pat), snip_type, text) in already_done:
                    continue

                analysis = categorize_with_llm(text)
                writer.writerow({
                    "patent_id": pat,
                    "snippet_type": snip_type,
                    "snippet_text": text,
                    **analysis,
                })
                f_out.flush()

                if analysis.get("category") == "ERROR":
                    consecutive_errors += 1
                    log.error(
                        f"Patent {pat} snippet failed: {analysis['reason']} "
                        f"({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} consecutive errors)"
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        log.error(f"Stopping: {MAX_CONSECUTIVE_ERRORS} consecutive errors. Resume with the same command.")
                        raise SystemExit(1)
                else:
                    consecutive_errors = 0
                    total_input_tokens += analysis.get("input_tokens", 0)
                    total_output_tokens += analysis.get("output_tokens", 0)

log.info(f"Run complete. Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")
