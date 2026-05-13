"""
Merge per-patent results from the three classification scripts into one summary table:
  - Patent_metadata.py / PatentClassification.py  → patent metadata (lens_id, title, ...)
  - Classify_Antibiotic_Snippets.py [+ CountABsnippets.py]  → antibiotic snippets + categories
  - PatentClassification.py  → end product / organism / sector

One row per patent. Raw claim/description text is excluded.
"""

import argparse
import os
import pandas as pd


def aggregate_snippets(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse snippet-level rows (one per snippet) into one row per patent.

    Per patent:
      - counts of snippets per category (BINGO, MARKER, AVOIDANCE, ...)
      - total snippet count
      - union of antibiotics found across all snippets (if `antibiotics_found` column exists)
    """
    df = df.copy()
    df["patent_id"] = df["patent_id"].astype(str)

    # Category counts: one column per category
    cat_counts = (
        df.groupby(["patent_id", "category"]).size()
        .unstack(fill_value=0)
        .add_prefix("n_")
        .reset_index()
    )
    cat_counts["n_snippets_total"] = cat_counts.drop(columns=["patent_id"]).sum(axis=1)

    # Union of antibiotics across snippets (only if CountABsnippets.py was run)
    if "antibiotics_found" in df.columns:
        def _union(values):
            seen = set()
            for v in values:
                if isinstance(v, str) and v:
                    seen.update(a.strip() for a in v.split(",") if a.strip())
            return ", ".join(sorted(seen))

        ab = (
            df.groupby("patent_id")["antibiotics_found"]
            .apply(_union)
            .reset_index()
            .rename(columns={"antibiotics_found": "antibiotics_found"})
        )
        ab["antibiotic_count"] = ab["antibiotics_found"].apply(
            lambda s: 0 if not s else len([x for x in s.split(",") if x.strip()])
        )
        cat_counts = cat_counts.merge(ab, on="patent_id", how="left")

    return cat_counts


def load_metadata(path: str) -> pd.DataFrame:
    """Read the patent metadata file. Accepts CSV or TSV based on extension."""
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    if "lens_id" in df.columns:
        df = df.rename(columns={"lens_id": "patent_id"})
    df["patent_id"] = df["patent_id"].astype(str)
    # Drop any free-text columns we don't want in the summary
    drop = [c for c in ("claims", "description", "text", "snippet_text") if c in df.columns]
    return df.drop(columns=drop)


def load_classification(path: str) -> pd.DataFrame:
    """Read the per-patent general classification (PatentClassification.py output)."""
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"lens_id": "patent_id"})
    df["patent_id"] = df["patent_id"].astype(str)
    keep = ["patent_id", "End_Product", "Organism", "Product_category", "Sector", "reason"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    if "reason" in out.columns:
        out = out.rename(columns={"reason": "classification_reason"})
    return out


def load_snippets(path: str) -> pd.DataFrame:
    """Read snippet-level antibiotic classifications and aggregate per patent."""
    df = pd.read_csv(path, sep="\t")
    return aggregate_snippets(df)


def main():
    parser = argparse.ArgumentParser(
        description="Merge metadata, antibiotic snippet classifications, and general "
                    "patent classification into one summary table (one row per patent)."
    )
    parser.add_argument("--metadata", help="Patent metadata CSV/TSV (lens_id, title, ...).")
    parser.add_argument("--snippets", help="Antibiotic snippet classification TSV "
                                            "(output of Classify_Antibiotic_Snippets.py, "
                                            "optionally enriched by CountABsnippets.py).")
    parser.add_argument("--classification", help="General patent classification TSV "
                                                 "(output of PatentClassification.py).")
    parser.add_argument("--output", required=True, help="Path to write the merged summary (CSV or TSV).")
    parser.add_argument(
        "--how",
        choices=["outer", "inner", "left"],
        default="outer",
        help="Join strategy across input tables (default: outer — keep all patents from any source).",
    )
    args = parser.parse_args()

    if not (args.metadata or args.snippets or args.classification):
        parser.error("Provide at least one of --metadata, --snippets, --classification.")

    frames = []
    if args.metadata:
        frames.append(("metadata", load_metadata(args.metadata)))
    if args.snippets:
        frames.append(("snippets", load_snippets(args.snippets)))
    if args.classification:
        frames.append(("classification", load_classification(args.classification)))

    print(f"Loaded sources: {[name for name, _ in frames]}")
    for name, df in frames:
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")

    merged = frames[0][1]
    for _, df in frames[1:]:
        merged = merged.merge(df, on="patent_id", how=args.how)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    sep = "\t" if args.output.lower().endswith(".tsv") else ","
    merged.to_csv(args.output, sep=sep, index=False)
    print(f"Wrote {len(merged)} rows × {len(merged.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
