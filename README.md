# LLM Deduplication Judge

A simple LLM-as-a-Judge pipeline for determining whether information
in candidate records is already covered by baseline records.

## Input

Two CSV files:

- `baseline.csv`
- `candidate.csv`

Both contain:

- `oblgr_nm`
- `url`
- `content_summary`

## Logic

For each candidate record:

1. Find baseline records with the same `oblgr_nm`.
2. If the URL exactly matches a baseline URL, mark it as contained
   without calling the LLM.
3. Otherwise, use an LLM to judge whether the candidate information
   is semantically contained in the baseline summaries.

## Installation

```bash
pip install -r requirements.txt