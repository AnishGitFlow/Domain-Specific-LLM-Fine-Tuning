"""Dataset preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from src.data.formatting import build_prompt, build_training_text
from src.utils.config import ensure_dir


@dataclass(frozen=True)
class PreparedDatasetPaths:
    train: Path
    validation: Path
    test: Path
    sample: Path


def normalize_row(row: dict, columns: dict[str, str]) -> dict[str, str]:
    """Normalize a raw dataset row into project fields."""
    raw_instruction = str(row.get(columns["instruction"], "")).strip()
    raw_input = str(row.get(columns["input"], "") or "").strip()
    output = str(row.get(columns["output"], "")).strip()
    question, context = extract_question_context(raw_instruction, raw_input)

    return {
        "instruction": question,
        "input": context,
        "question": question,
        "context": context,
        "output": output,
        "prompt": build_prompt(question, context),
        "completion": output,
        "text": build_training_text(question, context, output),
    }


def extract_question_context(raw_instruction: str, raw_input: str) -> tuple[str, str]:
    """Extract clean question and abstract context from PubMedQA-style rows."""
    context_prefix = "Answer the question based on the following context:"
    question_prefix = "Question:"

    context = raw_instruction
    if context.startswith(context_prefix):
        context = context[len(context_prefix) :].strip()

    question = raw_input
    if question.startswith(question_prefix):
        question = question[len(question_prefix) :].strip()

    return question, context


def quality_filter(row: dict) -> bool:
    """Keep useful non-empty supervised examples."""
    return bool(row["instruction"] and row["output"] and len(row["output"]) >= 3)


def prepare_dataset(config: dict) -> DatasetDict:
    """Load, normalize, deduplicate, and split a Hugging Face dataset."""
    dataset = load_dataset(config["dataset_name"], split=config.get("dataset_split", "train"))
    columns = config["text_columns"]

    normalized = dataset.map(
        lambda row: normalize_row(row, columns),
        remove_columns=dataset.column_names,
        desc="Formatting instruction samples",
    )
    normalized = normalized.filter(quality_filter, desc="Filtering low-quality samples")

    frame = normalized.to_pandas()
    frame = frame.drop_duplicates(subset=["instruction", "input", "output"]).reset_index(drop=True)
    normalized = Dataset.from_pandas(frame, preserve_index=False)

    splits = config["splits"]
    seed = int(config.get("seed", 42))
    test_size = float(splits["validation"]) + float(splits["test"])
    first_split = normalized.train_test_split(test_size=test_size, seed=seed)

    relative_test_size = float(splits["test"]) / test_size
    second_split = first_split["test"].train_test_split(test_size=relative_test_size, seed=seed)

    dataset_dict = DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )

    max_samples = config.get("max_samples") or {}
    for split_name, limit in max_samples.items():
        if limit:
            dataset_dict[split_name] = dataset_dict[split_name].select(
                range(min(int(limit), len(dataset_dict[split_name])))
            )

    return dataset_dict


def save_dataset(dataset: DatasetDict, output_dir: str | Path = "data/processed") -> PreparedDatasetPaths:
    """Save prepared splits as JSONL files."""
    output_path = ensure_dir(output_dir)
    sample_path = ensure_dir("data/samples")

    paths = PreparedDatasetPaths(
        train=output_path / "train.jsonl",
        validation=output_path / "validation.jsonl",
        test=output_path / "test.jsonl",
        sample=sample_path / "sample.jsonl",
    )

    dataset["train"].to_json(str(paths.train), orient="records", lines=True)
    dataset["validation"].to_json(str(paths.validation), orient="records", lines=True)
    dataset["test"].to_json(str(paths.test), orient="records", lines=True)
    dataset["train"].select(range(min(10, len(dataset["train"])))).to_json(
        str(paths.sample), orient="records", lines=True
    )

    return paths
