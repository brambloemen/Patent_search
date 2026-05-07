# Patent Search — Antibiotic Resistance Marker Analysis in Food & Feed Patents

A pipeline for identifying and classifying antibiotic resistance (AR) marker usage in biotechnology patents related to food, feed, vitamin, supplement, and enzyme production.

## Overview

This project screens patents in CPC classifications **C12N** (microbiology/genetics), **A21** (bakery), and **A23** (food/feed) for mentions of antibiotics. Each mention is extracted with surrounding context and classified by an LLM into categories reflecting how the antibiotic is used — as a selection marker in a production strain, in a marker-avoidance system, in a eukaryotic context, or irrelevant noise.

Patent data is sourced from [Lens.org](https://www.lens.org/). Antibiotic names are derived from the [CARD Antibiotic Resistance Ontology](https://card.mcmaster.ca/) (ARO).

## Workflow

```
Lens API export (JSON)
        │
        ▼
Patent_metadata.py          → metadata CSV (lens_id, title, word counts)
        │
        ▼
CARD_aro_ontology.py        → antibiotics_list.txt  (672 antibiotic names)
        │
        ▼
LLM_patent_class.py         → snippet extraction + LLM classification (TSV)
  ├─ FlashText keyword match
  ├─ Context window extraction (±300 chars)
  └─ Nebius gpt-oss-20b classification
        │
        ▼
CountABsnippets.py          → annotated results with per-snippet antibiotic counts
```

## Repository Structure

```
Patent_search/
├── scripts/
│   ├── CARD_aro_ontology.py       # Extract antibiotic names from CARD ARO
│   ├── Patent_metadata.py         # Extract patent metadata from Lens export
│   ├── LLM_patent_class.py        # Main pipeline: keyword match + LLM classify
│   └── CountABsnippets.py         # Annotate results with antibiotic counts
├── CARD_ontology/
│   ├── aro.json / aro.obo         # Antibiotic Resistance Ontology (CARD 2023)
│   ├── mo.json, ro.json           # Model and Relationship Ontologies
│   ├── ncbi_taxonomy.json         # NCBI Taxonomy
│   └── antibiotics_list.txt       # Derived list of 672 antibiotic names
├── results/                        # Pipeline output CSVs
├── envs/
│   └── patent.yaml                # Conda environment
└── .env                           # NEBIUS_API_KEY (not committed)
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
```

The classification script uses the [Nebius API](https://nebius.com/) with an OpenAI-compatible interface to run `gpt-oss-20b`.

**3. Obtain input data**

Export patents from [Lens.org](https://www.lens.org/) filtered by CPC codes `C12N`, `A21`, and `A23`. Save the export as `Lens_export_cpc_C12N_A21_A23.json`. The downstream scripts expect a pre-filtered JSONL file (`claims_FoodFeedSuppVitEnz_C12N.json`) with one patent record per line.

## Running the Pipeline

Run scripts in order from the `scripts/` directory:

```bash
# 1. Build antibiotic name list from CARD ontology
python CARD_aro_ontology.py

# 2. Extract patent metadata
python Patent_metadata.py

# 3. Classify antibiotic mentions with LLM (main step — expensive)
python LLM_patent_class.py

# 4. Annotate results with per-snippet antibiotic counts
python CountABsnippets.py
```

> **Cost warning**: `LLM_patent_class.py` estimates total token usage before running and prompts for confirmation. The dataset is large (~1.5 GB of patent text), so review the estimate before proceeding.

## Classification Categories

Each snippet (antibiotic mention + ±300 character context) is assigned one of:

| Category | Description |
|---|---|
| `BINGO` | Antibiotic resistance used as a marker in a food/feed production strain |
| `MARKER` | Antibiotic used as a bacterial selection marker (general) |
| `AVOIDANCE` | Marker-free system, marker removal, or antibiotic susceptibility context |
| `MARKER_AVOIDANCE` | Snippet covers both marker use and avoidance aspects |
| `EUKARYOTIC` | Antibiotic used in a eukaryotic (non-bacterial) context |
| `UNKNOWN` | Irrelevant context, general chemical list, or unclear usage |

## Key Dependencies

| Library | Purpose |
|---|---|
| `flashtext` | Fast multi-keyword extraction from large text |
| `obonet` / `networkx` | Parse and traverse CARD OBO ontology |
| `openai` | Client for Nebius API (OpenAI-compatible) |
| `pandas` | Data manipulation and CSV/TSV I/O |
| `python-dotenv` | Load API keys from `.env` |

## Data Sources

- **Patent data**: [Lens.org](https://www.lens.org/) — open patent metadata and full text
- **Antibiotic ontology**: [CARD 2023](https://card.mcmaster.ca/) — Comprehensive Antibiotic Resistance Database, licensed CC BY 4.0

## License

Apache License 2.0 — see [LICENSE](LICENSE).