# Patent Search — Antibiotic Resistance Marker Analysis in Food & Feed Patents

A pipeline for identifying and classifying antibiotic resistance (AR) marker usage in biotechnology patents related to food, feed, vitamin, supplement, and enzyme production.

## Overview

This project screens patents (obtained from lens.org) for mentions of antibiotics. Each mention is extracted with surrounding context and classified by an LLM into categories reflecting how the antibiotic is used - as a selection marker in a production strain, in a marker-avoidance system, in a eukaryotic context, or noise.

A second, more general script (`PatentClassification.py`) classifies patents by end product, production organism, product category, and sector without any antibiotic-specific filtering.

Patent data is sourced from [Lens.org](https://www.lens.org/). Antibiotic names are derived from the [CARD Antibiotic Resistance Ontology](https://card.mcmaster.ca/) (ARO).

## Workflow

```
Lens API export (JSONL)
        │
        ├──────────────────────────────────────────────────────┐
        ▼                                                       ▼
CARD_aro_ontology.py                               PatentClassification.py
→ antibiotics_list.txt (672 antibiotic names)      → end product / organism /
        │                                            sector classification (TSV)
        ▼
Classify_Antibiotic_Snippets.py
  ├─ FlashText keyword match
  ├─ Context window extraction (±300 chars)
  └─ LLM classification of each snippet (TSV)
        │
        ▼
CountABsnippets.py
→ annotated results with per-snippet antibiotic counts
```

## Repository Structure

```
Patent_search/
├── scripts/
│   ├── CARD_aro_ontology.py              # Extract antibiotic names from CARD ARO
│   ├── Patent_metadata.py                # Extract patent metadata from Lens export
│   ├── Classify_Antibiotic_Snippets.py   # Antibiotic mention detection + LLM classification
│   ├── PatentClassification.py           # General patent product/sector classification
│   └── CountABsnippets.py                # Annotate snippet results with antibiotic counts
├── CARD_ontology/
│   ├── aro.json / aro.obo                # Antibiotic Resistance Ontology (CARD 2023)
│   ├── mo.json, ro.json                  # Model and Relationship Ontologies
│   ├── ncbi_taxonomy.json                # NCBI Taxonomy
│   └── antibiotics_list.txt             # Derived list of 672 antibiotic names
├── results/                              # Pipeline output TSVs/CSVs
├── envs/
│   └── patent.yaml                       # Conda environment
└── .env                                  # API keys (not committed)
```

## Setup

**1. Create the conda environment**

```bash
conda env create -f envs/patent.yaml
conda activate patent
```

**2. Configure API access**

Create a `.env` file in the repository root:

```
NEBIUS_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here   # only needed for --provider google
```

**3. Obtain input data**

Export patents from [Lens.org](https://www.lens.org/) for your query of interest (e.g. filtered by CPC codes `C12N`, `A21`, and `A23`). The classification scripts expect a JSONL file with one Lens.org patent record per line.

Download the antibiotic ontology from [CARD](https://card.mcmaster.ca/download) and extract the contents into a `CARD_ontology/` directory.

## Running the Pipeline

Run scripts from the `scripts/` directory.

**1. Build the antibiotic name list** (once, from CARD ontology)

```bash
python CARD_aro_ontology.py
```

**2. Classify antibiotic snippets** (main antibiotic-focused pipeline)

```bash
python Classify_Antibiotic_Snippets.py --input ../path/to/patents.jsonl
```

Optional flags:

```
--provider  nebius|google        LLM provider (default: nebius)
--model     MODEL_ID             Override the default model for the provider
--antibiotic-list  PATH          Path to antibiotic name list (default: ../CARD_ontology/antibiotics_list.txt)
--window    INT                  Context window in characters around each match (default: 300)
--max-retries  INT               Retries per snippet on transient errors (default: 15)
```

Output is written incrementally to `results/<input_basename>_snip_class_<model>.tsv`. The script can be safely interrupted and resumed with the same command — already-classified snippets are skipped.

> **Cost note**: the script logs estimated token usage. For large datasets, run on a subset of patents first to estimate costs to avoid unexpected bills or rate limits.

**3. Annotate results with antibiotic counts**

```bash
python CountABsnippets.py
```

**4. General patent classification** (product, organism, sector — no antibiotic filter)

```bash
python PatentClassification.py --input ../path/to/patents.jsonl
```

Optional flags:

```
--provider  nebius|google        LLM provider (default: nebius)
--model     MODEL_ID             Override the default model for the provider
--max-retries  INT               Retries per patent on transient errors (default: 15)
```

Output: `results/<input_basename>_claims_classifications_<model>.tsv`

## Classification Categories

### Classify_Antibiotic_Snippets.py

Each snippet (antibiotic mention + ±300 character context) is assigned one of:

| Category | Description |
|---|---|
| `BINGO` | Antibiotic resistance used as a marker in a food/feed production strain |
| `MARKER` | Antibiotic used as a bacterial selection marker (general) |
| `AVOIDANCE` | Marker-free system, marker removal, or antibiotic susceptibility context |
| `MARKER_AVOIDANCE` | Snippet covers both marker use and avoidance aspects |
| `EUKARYOTIC` | Antibiotic used in a eukaryotic (non-bacterial) context |
| `UNKNOWN` | Irrelevant context, general chemical list, or unclear usage |

### PatentClassification.py

Each patent's claims and description are classified with four fields:

| Field | Values |
|---|---|
| `End_Product` | Free text (e.g. "lysine", "amylase") |
| `Organism` | Plant / bacterium / yeast / animal / human / unknown |
| `Product_category` | Amino acid, oligosaccharide, vitamin, food colour/flavour, enzyme, peptide, vector, other |
| `Sector` | food/feed, medicinal, diagnostic, molecular biology, chemistry, other |

## LLM Providers

Both classification scripts support two providers via `--provider`:

| Provider | Default model | Env var |
|---|---|---|
| `nebius` (default) | `openai/gpt-oss-20b` | `NEBIUS_API_KEY` |
| `google` | `gemma-4-31b-it` | `GOOGLE_API_KEY` |

Any model available on the chosen provider can be selected with `--model`. The Nebius provider uses an OpenAI-compatible API, so any OpenAI-compatible endpoint can be targeted by pointing at the right base URL in the script.

## Key Dependencies

| Library | Purpose |
|---|---|
| `flashtext` | Fast multi-keyword extraction from large text |
| `obonet` / `networkx` | Parse and traverse CARD OBO ontology |
| `openai` | Client for Nebius/OpenAI-compatible APIs |
| `google-genai` | Client for Google AI Studio |
| `pandas` | Data manipulation and TSV/CSV I/O |
| `python-dotenv` | Load API keys from `.env` |

## Data Sources

- **Patent data**: [Lens.org](https://www.lens.org/) — open patent metadata and full text
- **Antibiotic ontology**: [CARD 2023](https://card.mcmaster.ca/) — Comprehensive Antibiotic Resistance Database, licensed CC BY 4.0

## License

Apache License 2.0 — see [LICENSE](LICENSE).
