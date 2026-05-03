"""Create clean Streamlit demo examples from the processed test split."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline import extract_question_context


def main() -> None:
    input_path = Path("data/processed/test.jsonl")
    output_path = Path("docs/demo_examples.md")

    rows = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            question = row.get("question")
            context = row.get("context")
            if not question or not context:
                question, context = extract_question_context(row["instruction"], row["input"])
            rows.append(
                {
                    "question": question,
                    "context": context,
                    "reference": row["output"],
                }
            )
            if len(rows) == 5:
                break

    lines = ["# Streamlit Demo Examples", ""]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Example {index}",
                "",
                "**Question**",
                "",
                row["question"],
                "",
                "**Context**",
                "",
                row["context"],
                "",
                "**Reference Answer**",
                "",
                row["reference"],
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
