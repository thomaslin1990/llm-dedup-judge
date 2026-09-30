import os
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

BASELINE_FILE = "baseline.csv"
CANDIDATE_FILE = "candidate.csv"
OUTPUT_FILE = "candidate_judged.csv"

MODEL = "gpt-6-luna"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------
# Structured output
# ---------------------------------------------------------

class JudgeResult(BaseModel):
    is_contained: bool
    matched_baseline_row: Optional[int]
    reason: str


# ---------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------

def judge_candidate(
    oblgr_nm: str,
    candidate_summary: str,
    baseline_records: pd.DataFrame,
) -> JudgeResult:

    if baseline_records.empty:
        return JudgeResult(
            is_contained=False,
            matched_baseline_row=None,
            reason="No baseline records found for this obligor."
        )

    baseline_text = []

    for idx, row in baseline_records.iterrows():
        baseline_text.append(
            f"""
BASELINE_ROW_ID: {idx}
SUMMARY:
{row['content_summary']}
""".strip()
        )

    baseline_text = "\n\n---\n\n".join(baseline_text)

    prompt = f"""
You are an information deduplication judge.

Your task is to determine whether the information in a CANDIDATE summary
is already contained in any of the BASELINE summaries.

Obligor:
{oblgr_nm}

CANDIDATE:
{candidate_summary}

BASELINE RECORDS:
{baseline_text}

Judging rules:

1. Judge semantic information, not wording.
2. If a baseline record describes the same underlying event/fact as the
   candidate, consider the candidate contained.
3. Small wording differences or different levels of summarization do not
   make something new.
4. Merely discussing the same company or same general topic is NOT enough.
5. Different events, dates, amounts, transactions, announcements, or
   developments should be considered different information.
6. If the candidate contains a materially new fact that is not present
   in the baseline record, mark it as NOT contained.
7. Set matched_baseline_row to the BASELINE_ROW_ID of the best matching
   baseline record.
8. If no baseline record contains the candidate information,
   matched_baseline_row must be null.

Return your judgment.
"""

    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict semantic deduplication judge. "
                    "Avoid false positive matches."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=JudgeResult,
    )

    return response.output_parsed


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    baseline = pd.read_csv(BASELINE_FILE)
    candidate = pd.read_csv(CANDIDATE_FILE)

    # -----------------------------------------------------
    # Basic cleanup
    # -----------------------------------------------------

    for df in [baseline, candidate]:
        df["oblgr_nm"] = df["oblgr_nm"].fillna("").str.strip()
        df["url"] = df["url"].fillna("").str.strip()
        df["content_summary"] = df["content_summary"].fillna("").str.strip()

    # -----------------------------------------------------
    # Group baseline by obligor
    # -----------------------------------------------------

    baseline_groups = {
        name: group
        for name, group in baseline.groupby("oblgr_nm")
    }

    results = []

    # -----------------------------------------------------
    # Judge each candidate row
    # -----------------------------------------------------

    for i, row in candidate.iterrows():

        oblgr_nm = row["oblgr_nm"]
        candidate_url = row["url"]
        candidate_summary = row["content_summary"]

        print(
            f"[{i + 1}/{len(candidate)}] "
            f"Judging {oblgr_nm}"
        )

        baseline_records = baseline_groups.get(
            oblgr_nm,
            pd.DataFrame()
        )

        # =================================================
        # Shortcut 1:
        # No baseline records for this obligor
        # =================================================

        if baseline_records.empty:

            results.append({
                "is_contained": False,
                "match_type": "no_baseline",
                "matched_baseline_row": None,
                "matched_baseline_url": None,
                "reason": "No baseline records found for this obligor.",
            })

            continue

        # =================================================
        # Shortcut 2:
        # Exact URL already exists in baseline
        # =================================================

        if candidate_url:

            url_matches = baseline_records[
                baseline_records["url"] == candidate_url
            ]

            if not url_matches.empty:

                matched_idx = url_matches.index[0]

                print("  -> URL exact match, skip LLM")

                results.append({
                    "is_contained": True,
                    "match_type": "url_exact_match",
                    "matched_baseline_row": matched_idx,
                    "matched_baseline_url": candidate_url,
                    "reason": "Exact URL already exists in baseline.",
                })

                continue

        # =================================================
        # Otherwise: use LLM
        # =================================================

        try:

            print("  -> Calling LLM")

            judgment = judge_candidate(
                oblgr_nm=oblgr_nm,
                candidate_summary=candidate_summary,
                baseline_records=baseline_records,
            )

            matched_url = None

            if judgment.matched_baseline_row is not None:

                matched_id = judgment.matched_baseline_row

                if matched_id in baseline.index:
                    matched_url = baseline.loc[
                        matched_id,
                        "url"
                    ]

            results.append({
                "is_contained": judgment.is_contained,
                "match_type": "llm",
                "matched_baseline_row": judgment.matched_baseline_row,
                "matched_baseline_url": matched_url,
                "reason": judgment.reason,
            })

        except Exception as e:

            print(f"  -> ERROR: {e}")

            results.append({
                "is_contained": None,
                "match_type": "error",
                "matched_baseline_row": None,
                "matched_baseline_url": None,
                "reason": f"ERROR: {e}",
            })

    # -----------------------------------------------------
    # Combine results
    # -----------------------------------------------------

    result_df = pd.concat(
        [
            candidate.reset_index(drop=True),
            pd.DataFrame(results),
        ],
        axis=1,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\nDone.")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nMatch type summary:")
    print(result_df["match_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()